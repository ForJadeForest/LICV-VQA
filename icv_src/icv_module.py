import hydra
import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
from loguru import logger
from torch import optim
from transformers import get_cosine_schedule_with_warmup

from lmm_icl_interface import LMMInterface

from .icv_encoder.global_icv_encoder import GlobalICVEncoder
from .icv_model.icv_intervention import LearnableICVInterventionLMM


class VQAICVModule(pl.LightningModule):
    def __init__(
        self,
        interface: LMMInterface,
        module_cfg,
        lmm_cfg,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["interface"])
        self.module_cfg = module_cfg
        self.lmm_cfg = lmm_cfg
        self.interface = interface

        self.interface.requires_grad_(False)
        if hasattr(self.interface.model, "gradient_checkpointing_enable"):
            self.interface.model.gradient_checkpointing_enable()

        self.icv_model = LearnableICVInterventionLMM(
            interface,
            enable_intervention=True,
            intervention_layer=self.lmm_cfg.intervention_layer,
            layer_format=self.lmm_cfg.layer_format,
            total_layers=self.lmm_cfg.total_layers,
        )

        icv_encoder_factor: GlobalICVEncoder = hydra.utils.instantiate(
            module_cfg.icv_encoder, _partial_=True
        )
        icv_layer_num = len(self.icv_model.intervention_layer_names)
        hidden_dim = self.lmm_cfg.hidden_size
        self.icv_encoder = icv_encoder_factor(
            lmm_hidden_dim=hidden_dim, lmm_layers=icv_layer_num
        )

        self.temperature = torch.nn.Parameter(
            torch.tensor(self.module_cfg.init_temperature),
            requires_grad=module_cfg.learnable_t,
        )

    def on_train_start(self) -> None:
        if self.module_cfg.decay_per_step < 0:
            return -1
        if isinstance(self.module_cfg.decay_per_step, int):
            decay_per_step = self.module_cfg.decay_per_step
        elif (
            isinstance(self.module_cfg.decay_per_step, float)
            and 0 < self.module_cfg.decay_per_step < 1
        ):
            decay_per_step = int(
                self.trainer.estimated_stepping_batches * self.module_cfg.decay_per_step
            )
        else:
            raise ValueError("decay_ratio must be an int or a float between 0 and 1")
        self.decay_per_step = decay_per_step
        return super().on_train_start()

    def forward(
        self,
        query_inputs,
        inputs,
        query_x_length,
        in_context_length,
    ) -> torch.Any:
        """
        in_context_input: shot_num1 + ... + shot_num_n, seq_len
        ice_shot_num: bs, 1
        query_input: bs, query_seq_len
        inputs: bs, ice_seq_len + query_seq_len,
        """
        icl_context_mask = self.get_mask(inputs, in_context_length)
        zero_shot_mask = self.get_mask(query_inputs, query_x_length)

        icv_encoder_output = self.icv_encoder()

        icv = (
            icv_encoder_output.alpha.unsqueeze(dim=-1)
            * icv_encoder_output.in_context_vector
        )

        if self.module_cfg.hard_loss_weight:
            query_inputs["labels"] = query_inputs["input_ids"]

        self.icv_model.toggle_intervention(True)
        icv_outputs = self.icv_model(**query_inputs, icv=icv)

        if self.module_cfg.only_hard_loss:
            return {"loss": icv_outputs["loss"]}, icv_encoder_output
        icv_logits = icv_outputs["logits"]
        with torch.no_grad():
            self.icv_model.toggle_intervention(False)
            ice_logits = self.icv_model(**inputs)["logits"]

        loss = 0.0
        kl_loss = self.calculate_kl_divergence(
            icv_logits[zero_shot_mask].view(-1, icv_logits.shape[-1]),
            ice_logits[icl_context_mask].view(-1, ice_logits.shape[-1]),
        )
        loss += kl_loss

        loss_dict = {"kl_loss": kl_loss}
        if self.module_cfg.hard_loss_weight:
            loss += self.module_cfg.hard_loss_weight * icv_outputs["loss"]
            loss_dict["ce_loss"] = icv_outputs["loss"]
        loss_dict["loss"] = loss
        return loss_dict, icv_encoder_output

    def calculate_kl_divergence(self, stu_logits, tea_logits):
        stu_logits /= self.temperature
        tea_logits /= self.temperature
        return (
            (
                tea_logits.softmax(dim=1)
                * (
                    (tea_logits.softmax(dim=1) + self.module_cfg.kl_eps).log()
                    - (stu_logits.softmax(dim=1) + self.module_cfg.kl_eps).log()
                )
            )
            .sum(dim=1)
            .mean()
        ) * self.temperature**2

    def get_mask(self, inputs, mask_length):
        mask_shape = inputs[self.interface.input_ids_field_name].shape
        bs, seq_len = mask_shape
        device = inputs[self.interface.input_ids_field_name].device
        sequence_indices = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)
        )
        mask = sequence_indices >= mask_length.unsqueeze(dim=1)
        mask[
            inputs[self.interface.input_ids_field_name]
            == self.interface.tokenizer.pad_token_id
        ] = False
        return mask

    def decay_temperature(self):
        if self.module_cfg.decay_ratio < 0:
            return

        if self.global_step % self.decay_per_step == 0 and self.global_step != 0:
            self.temperature = torch.clip(
                self.temperature * self.module_cfg.decay_ratio,
                min=self.module_cfg.min_tmeprature,
            )

    def training_step(self, batch, batch_idx):
        self.decay_temperature()
        loss_dict, icv_encoder_output = self(**batch)
        self.log_dict(loss_dict, sync_dist=True, prog_bar=True)
        if self.module_cfg.log_alpha:
            alpha = icv_encoder_output.alpha.squeeze()
            for i in range(len(alpha)):
                self.log(f"alpha/alpha-{i}", alpha[i])
        self.log("temperature", self.temperature)
        return loss_dict["loss"]

    def configure_optimizers(self):
        params = []
        for name, param in self.icv_encoder.named_parameters():
            if not param.requires_grad:
                continue
            if "alpha" in name:
                params.append({"params": param, "lr": self.module_cfg.alpha_lr})
            else:
                params.append({"params": param})

        if "deepspeed" in self.module_cfg.strategy:
            optimizer = DeepSpeedCPUAdam(
                params,
                lr=self.module_cfg.icv_lr,
                weight_decay=self.module_cfg.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                params,
                lr=self.module_cfg.icv_lr,
                weight_decay=self.module_cfg.weight_decay,
            )

        step_batches = self.trainer.estimated_stepping_batches
        if isinstance(self.module_cfg.warm_steps, float):
            warm_steps = self.module_cfg.warm_steps * step_batches
        elif isinstance(self.module_cfg.warm_steps, int):
            warm_steps = self.module_cfg.warm_steps
        else:
            raise ValueError(
                f"the warm_steps should be int or float, but got {type(self.module_cfg.warm_steps)}"
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_steps, num_training_steps=step_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint):
        # Only save the icv weights
        params_name = list(checkpoint["state_dict"].keys())
        for name in params_name:
            if name.startswith("model"):
                checkpoint["state_dict"].pop(name)

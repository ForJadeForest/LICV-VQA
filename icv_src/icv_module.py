import hydra
import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
from loguru import logger
from torch import optim
from transformers import get_cosine_schedule_with_warmup

from .icv_encoder.global_icv_encoder import GlobalICVEncoder
from .icv_model import ICVIdeficsForVisionText2Text


class VQAICVModule(pl.LightningModule):
    def __init__(
        self,
        model: ICVIdeficsForVisionText2Text,
        processor,
        module_cfg,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "processor"])
        self.model = model
        self.processor = processor

        self.model.requires_grad_(False)
        self.model.gradient_checkpointing_enable()

        self.module_cfg = module_cfg
        icv_encoder_factor: GlobalICVEncoder = hydra.utils.instantiate(
            module_cfg.icv_encoder, _partial_=True
        )
        layers = self.model.config.num_hidden_layers
        hidden_dim = self.model.config.hidden_size
        self.icv_encoder = icv_encoder_factor(
            llm_hidden_dim=hidden_dim, llm_layers=layers
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
        query_input,
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
        # - 1 为了去除query中的bos
        original_mask_length = query_x_length + in_context_length - 1
        original_mask = self.get_mask(inputs, original_mask_length)
        query_mask = self.get_mask(query_input, query_x_length)

        icv_encoder_output = self.icv_encoder()
        labels = None
        if self.module_cfg.hard_loss_weight:
            labels = query_input["input_ids"]

        icv_outputs = self.model(
            **query_input,
            in_context_vector=icv_encoder_output.in_context_vector,
            alpha=icv_encoder_output.alpha,
            labels=labels,
        )
        if self.module_cfg.only_hard_loss:
            return {"loss": icv_outputs["loss"]}, icv_encoder_output
        icv_logits = icv_outputs["logits"]
        with torch.no_grad():
            ice_logits = self.model(**inputs)["logits"]

        loss = 0.0
        kl_loss = self.calculate_kl_divergence(
            icv_logits[query_mask].view(-1, icv_logits.shape[-1]),
            ice_logits[original_mask].view(-1, ice_logits.shape[-1]),
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
        mask_shape = inputs["input_ids"].shape
        bs, seq_len = mask_shape
        device = inputs["input_ids"].device
        sequence_indices = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)
        )
        mask = sequence_indices >= mask_length.unsqueeze(dim=1)
        mask[inputs.input_ids == self.processor.tokenizer.pad_token_id] = False
        return mask

    def decay_temperature(self):
        # 判断ratio的类型并计算衰减的步数间隔
        if self.module_cfg.decay_ratio < 0:
            return

        # 如果当前步数达到衰减步数，进行衰减
        if self.global_step % self.decay_per_step == 0 and self.global_step != 0:
            self.temperature = torch.clip(
                self.temperature * self.module_cfg.decay_ratio,
                min=self.module_cfg.min_tmeprature,
            )

    def training_step(self, batch, batch_idx):
        self.decay_temperature()
        loss_dict, icv_encoder_output = self(**batch)
        self.log_dict(loss_dict, sync_dist=True)
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
        params_name = list(checkpoint["state_dict"].keys())
        for name in params_name:
            if name.startswith("model"):
                checkpoint["state_dict"].pop(name)

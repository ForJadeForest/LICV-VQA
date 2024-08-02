import os
import shutil
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    IdeficsProcessor,
    get_cosine_schedule_with_warmup,
    IdeficsForVisionText2Text,
)


from icv_src.icv_datasets.load_ds_utils import (
    load_coco_ds,
    load_okvqa_ds,
    load_vqav2_ds,
)

from lmm_icl_interface import (
    LMMPromptManager,
    LMMPromptProcessor,
    Idefics2PromptProcessor,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VQADataset(Dataset):
    def __init__(
        self,
        name,
        root_dir,
        train_coco_dataset_root,
        val_coco_dataset_root,
        prompt_manager: LMMPromptManager,
        max_train_size=10000,
        split="train",
        val_ann_file=None,
    ):
        super().__init__()
        self.prompt_manager = prompt_manager
        if name == "vqav2":
            ds = load_vqav2_ds(
                root_dir,
                train_coco_dataset_root,
                val_coco_dataset_root,
                split=split,
                val_ann_file=val_ann_file,
            )
        elif name == "okvqa":
            ds = load_okvqa_ds(
                root_dir,
                train_coco_dataset_root,
                val_coco_dataset_root,
                split=split,
            )
        self.query_ds = ds
        if max_train_size > 0 and len(self.query_ds) > max_train_size:
            random_select_idx = np.random.choice(
                len(self.query_ds), size=max_train_size, replace=False
            )
            self.query_ds = self.query_ds.select(random_select_idx)

    def __len__(self):
        return len(self.query_ds)

    def __getitem__(self, index):
        query_item = self.query_ds[index]

        query_prompt = [
            query_item["image"],
            self.prompt_manager.gen_ice_text_with_label(
                query_item, add_sep_token=False
            ),
        ]

        query_x = [
            query_item["image"],
            self.prompt_manager.gen_query_text_without_label(query_item),
        ]
        return {
            "prompt": query_prompt,
            "prompt_x": query_x,
        }


class CaptionDataset(Dataset):
    def __init__(
        self,
        name,
        train_coco_dataset_root,
        val_coco_dataset_root,
        train_coco_annotation_file,
        val_coco_annotation_file,
        prompt_manager: LMMPromptManager,
        max_train_size=8000,
        split="train",
    ):
        super().__init__()
        self.prompt_manager = prompt_manager
        if name == "coco2017":
            ds = load_coco_ds(
                train_coco_dataset_root=train_coco_dataset_root,
                train_coco_annotation_file=train_coco_annotation_file,
                val_coco_dataset_root=val_coco_dataset_root,
                val_coco_annotation_file=val_coco_annotation_file,
                split=split,
            )
        self.query_ds = ds

        if max_train_size > 0 and len(self.query_ds) > max_train_size:
            random_select_idx = np.random.choice(
                len(self.query_ds), size=max_train_size, replace=False
            )
            self.query_ds = self.query_ds.select(random_select_idx)

    def __len__(self):
        return len(self.query_ds)

    def __getitem__(self, index):
        query_item = self.query_ds[index]
        query_prompt = [
            query_item["image"],
            self.prompt_manager.gen_ice_text_with_label(
                query_item, add_sep_token=False
            ),
        ]

        query_x = [
            query_item["image"],
            self.prompt_manager.gen_query_text_without_label(query_item),
        ]
        return {
            "prompt": query_prompt,
            "prompt_x": query_x,
        }


class FTVQADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_cfg,
        processor,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="processor")
        self.processor = processor
        self.data_cfg = data_cfg
        self.collator_data = partial(collator_data, processor=self.processor)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self.data_cfg.task.task_name == "vqa":
                self.train_ds = VQADataset(
                    name=self.data_cfg.task.datasets.name,
                    root_dir=self.data_cfg.task.datasets.root_dir,
                    train_coco_dataset_root=self.data_cfg.task.datasets.train_coco_dataset_root,
                    val_coco_dataset_root=self.data_cfg.task.datasets.val_coco_dataset_root,
                    prompt_manager=self.prompt_manager,
                    max_train_size=self.data_cfg.task.datasets.max_train_size,
                    split="train",
                    val_ann_file=getattr(
                        self.data_cfg.task.datasets, "val_ann_file", None
                    ),
                )
            elif self.data_cfg.task.task_name == "caption":
                self.train_ds = CaptionDataset(
                    name=self.data_cfg.task.datasets.name,
                    train_coco_dataset_root=self.data_cfg.task.datasets.train_coco_dataset_root,
                    val_coco_dataset_root=self.data_cfg.task.datasets.val_coco_dataset_root,
                    train_coco_annotation_file=self.data_cfg.task.datasets.train_coco_annotation_file,
                    val_coco_annotation_file=self.data_cfg.task.datasets.val_coco_annotation_file,
                    prompt_manager=self.prompt_manager,
                    max_train_size=self.data_cfg.task.datasets.max_train_size,
                    split="train",
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            self.data_cfg.bs,
            num_workers=self.data_cfg.num_workers,
            collate_fn=self.collator_data,
        )


def collator_data(data_list, prompt_processor: LMMPromptProcessor):
    sample = data_list[0]
    data_dict = {k: [d[k] for d in data_list] for k in sample.keys()}
    query_prompt = data_dict["query_prompt"]
    query_x = data_dict["query_x"]

    query_input = prompt_processor.prepare_input(
        query_prompt,
        padding=True,
        truncation=True,
        add_eos_token=True,
    )
    query_x = prompt_processor.prepare_input(
        query_x, return_tensors="pt", padding=True, truncation=True
    )

    query_x_length = (
        query_x[prompt_processor.input_ids_field]
        != prompt_processor.tokenizer.pad_token_id
    ).sum(dim=1)
    return {
        "inputs": query_input,
        "query_x_length": query_x_length,
    }


class FTVQAModule(pl.LightningModule):
    def __init__(self, model, processor, module_cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "processor"])
        self.model = model
        self.processor = processor

        self.module_cfg = module_cfg

    def training_step(self, batch, batch_idx):
        if self.module_cfg.only_ans_loss:
            query_mask = self.get_mask(batch["inputs"], batch["query_x_length"])
            labels = torch.full_like(
                batch["inputs"]["input_ids"],
                device=batch["inputs"]["input_ids"].device,
                fill_value=-100,
            )
            labels[query_mask] = batch["inputs"]["input_ids"][query_mask]
        else:
            labels = batch["inputs"]["input_ids"]

        output = self.model(**batch["inputs"], labels=labels)
        self.log("loss", output["loss"], prog_bar=True)
        return output["loss"]

    def configure_optimizers(self):
        if "deepspeed" in self.module_cfg.strategy:
            optimizer = DeepSpeedCPUAdam(
                filter(lambda x: x.requires_grad, self.parameters()),
                lr=self.module_cfg.lr,
                weight_decay=self.module_cfg.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                filter(lambda x: x.requires_grad, self.parameters()),
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


@hydra.main(config_path="config", config_name="lora_train.yaml")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    if not os.path.exists(cfg.result_dir):
        os.makedirs(cfg.result_dir)

    model_name = cfg.model_name_or_path.split("/")[-1]
    save_path = Path(
        os.path.join(
            cfg.result_dir,
            "model_cpk",
            cfg.data_cfg.dataset.name,
            model_name,
            cfg.run_name,
        )
    )
    cpk_path = save_path / "last.ckpt"
    if cpk_path.exists():
        logger.info(f"The Checkpoint of {cfg.run_name} exists, EXIT....")
        return
    wb_logger = WandbLogger(
        save_dir=cfg.result_dir,
        name=cfg.run_name,
        project="VQAInContextVector",
        log_model=False,
    )
    wb_logger.log_hyperparams(cfg)

    trainer = pl.Trainer(
        logger=wb_logger,
        callbacks=[
            LearningRateMonitor(),
            RichModelSummary(max_depth=2),
            RichProgressBar(),
        ],
        enable_checkpointing=False,
        **cfg.trainer,
    )

    model = IdeficsForVisionText2Text.from_pretrained(cfg.model_name_or_path)
    processor = Idefics2PromptProcessor(cfg.model_name_or_path)
    processor.tokenizer.padding_side = "right"
    if cfg.lora_qkv:
        target_modules = ["q_proj", "v_proj", "k_proj"]
    else:
        target_modules = ["lm_head"]

    config = LoraConfig(
        r=32,
        lora_alpha=8,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    # get the peft model
    model = get_peft_model(model, config)
    if not cfg.lora_qkv:
        new_lora_B = nn.Linear(32, 32002, bias=False)
        nn.init.constant_(new_lora_B.weight, 0)
        model.lm_head.lora_B["default"] = new_lora_B
    model.print_trainable_parameters()
    module = FTVQAModule(
        model=model,
        processor=processor,
        module_cfg=cfg.module_cfg,
    )
    data_module = FTVQADataModule(data_cfg=cfg.data_cfg, processor=processor)

    trainer.fit(
        module,
        data_module,
    )
    trainer.save_checkpoint(
        filepath=cpk_path,
        weights_only=True,
    )


if __name__ == "__main__":
    load_dotenv(override=True)
    main()

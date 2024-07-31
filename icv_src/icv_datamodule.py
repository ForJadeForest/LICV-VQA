from functools import partial

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lmm_icl_interface import LMMPromptManager, LMMPromptProcessor

from .icv_datasets.vqa_dataset import VQADataset
from .icv_datasets.caption_dataset import CaptionDataset


class VQAICVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_cfg,
        prompt_manager: LMMPromptManager,
        prompt_processor: LMMPromptProcessor,
    ) -> None:
        super().__init__()

        self.prompt_processor = prompt_processor
        self.prompt_processor.tokenizer.padding_side = "right"
        self.prompt_manager = prompt_manager
        self.collator_data = partial(
            collator_data, prompt_processor=self.prompt_processor
        )
        self.data_cfg = data_cfg

    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self.data_cfg.task.task_name == "vqa":
                self.train_ds = VQADataset(
                    name=self.data_cfg.task.datasets.name,
                    root_dir=self.data_cfg.task.datasets.root_dir,
                    train_coco_dataset_root=self.data_cfg.task.datasets.train_coco_dataset_root,
                    val_coco_dataset_root=self.data_cfg.task.datasets.val_coco_dataset_root,
                    prompt_manager=self.prompt_manager,
                    instruction=self.data_cfg.task.instruction,
                    few_shot_num=self.data_cfg.task.datasets.few_shot_num,
                    max_train_size=self.data_cfg.task.datasets.max_train_size,
                    split="train",
                    val_ann_file=getattr(
                        self.data_cfg.task.datasets, "val_ann_file", None
                    ),
                    filter_ques_type=self.data_cfg.task.datasets.filter_ques_type,
                    select_from_query=self.data_cfg.task.datasets.select_from_query,
                )
            elif self.data_cfg.task.task_name == "caption":
                self.train_ds = CaptionDataset(
                    name=self.data_cfg.task.datasets.name,
                    train_coco_dataset_root=self.data_cfg.task.datasets.train_coco_dataset_root,
                    val_coco_dataset_root=self.data_cfg.task.datasets.val_coco_dataset_root,
                    train_coco_annotation_file=self.data_cfg.task.datasets.train_coco_annotation_file,
                    val_coco_annotation_file=self.data_cfg.task.datasets.val_coco_annotation_file,
                    prompt_manager=self.prompt_manager,
                    instruction=self.data_cfg.task.instruction,
                    few_shot_num=self.data_cfg.task.datasets.few_shot_num,
                    max_train_size=self.data_cfg.task.datasets.max_train_size,
                    split="train",
                    select_from_query=self.data_cfg.task.datasets.select_from_query,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            self.data_cfg.bs,
            num_workers=self.data_cfg.num_workers,
            collate_fn=self.collator_data,
            pin_memory=True,
        )


def collator_data(data_list, prompt_processor: LMMPromptProcessor):
    sample = data_list[0]
    data_dict = {k: [d[k] for d in data_list] for k in sample.keys()}
    query_prompt = data_dict["query_prompt"]
    ice_prompt = data_dict["ice_prompt"]
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
    ice_input = prompt_processor.prepare_input(
        ice_prompt, return_tensors="pt", padding=True, truncation=True
    )
    input_prompts = []

    for ice, query in zip(ice_prompt, query_prompt):
        input_prompts.append(ice + query)

    inputs = prompt_processor.prepare_input(
        input_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_eos_token=True,
    )
    in_context_length = (
        (
            ice_input[prompt_processor.input_ids_field]
            != prompt_processor.tokenizer.pad_token_id
        )
    ).sum(dim=1)
    query_x_without_bos_length = (
        (
            query_x[prompt_processor.input_ids_field]
            != prompt_processor.tokenizer.pad_token_id
        )
        & (
            query_x[prompt_processor.input_ids_field]
            != prompt_processor.tokenizer.bos_token_id
        )
    ).sum(dim=1)

    query_x_length = (
        query_x[prompt_processor.input_ids_field]
        != prompt_processor.tokenizer.pad_token_id
    ).sum(dim=1)
    return {
        "query_inputs": query_input,
        "inputs": inputs,
        "in_context_length": in_context_length + query_x_without_bos_length,
        "query_x_length": query_x_length,
    }

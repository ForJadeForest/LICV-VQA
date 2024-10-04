import numpy as np
from loguru import logger
from torch.utils.data import Dataset

from lmm_icl_interface import LMMPromptManager

from .load_ds_utils import load_coco_ds


class CaptionDataset(Dataset):
    def __init__(
        self,
        name,
        train_coco_dataset_root,
        val_coco_dataset_root,
        train_coco_annotation_file,
        val_coco_annotation_file,
        prompt_manager: LMMPromptManager,
        instruction="",
        few_shot_num=8,
        max_train_size=10000,
        split="train",
        select_from_query=True,
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
        if select_from_query:
            self.select_ds = self.query_ds
        else:
            self.select_ds = ds
        self.few_shot_num = few_shot_num
        self.instruction = instruction
        logger.info(
            f"Query dataset size: {len(self.query_ds)}, Select dataset size: {len(self.select_ds)}"
        )

    def __len__(self):
        return len(self.query_ds)

    def __getitem__(self, index):
        query_item = self.query_ds[index]

        few_shot_index = np.random.choice(
            len(self.select_ds), size=self.few_shot_num
        ).tolist()
        select_global_idx = self.select_ds[few_shot_index]["idx"]
        while query_item["idx"] in select_global_idx:
            few_shot_index = np.random.choice(
                len(self.select_ds), size=self.few_shot_num
            ).tolist()
            select_global_idx = self.select_ds[few_shot_index]["idx"]

        in_context_example = [self.select_ds[idx] for idx in few_shot_index]
        in_context_text = [
            [
                ice["image"],
                self.prompt_manager.gen_ice_text_with_label(ice, add_sep_token=True),
            ]
            for ice in in_context_example
        ]
        prompt = []
        if self.instruction:
            prompt = [self.instruction]
        for ic_prompt in in_context_text:
            prompt.extend(ic_prompt)

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
            "ice_prompt": prompt,
            "query_prompt": query_prompt,
            "query_x": query_x,
        }

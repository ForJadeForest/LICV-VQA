from torch.utils.data import Dataset
import numpy as np
from .load_ds_utils import load_okvqa_ds, load_vqav2_ds


class VQAV2Dataset(Dataset):
    def __init__(
        self,
        name,
        root_dir,
        train_coco_dataset_root,
        val_coco_dataset_root,
        instruction="",
        few_shot_num=8,
        max_train_size=10000,
        split="train",
        val_ann_file=None,
        filter_ques_type=None,
    ):
        super().__init__()
        if name == "vqav2":
            self.ds = load_vqav2_ds(
                root_dir,
                train_coco_dataset_root,
                val_coco_dataset_root,
                split=split,
                val_ann_file=val_ann_file,
                # filter_ques_type=filter_ques_type,
            )
        elif name == "okvqa":
            self.ds = load_okvqa_ds(
                root_dir,
                train_coco_dataset_root,
                val_coco_dataset_root,
                split=split,
                # filter_ques_type=filter_ques_type,
            )
        self.query_ds = self.ds
        if filter_ques_type:
            self.query_ds = self.ds.filter(
                lambda x: [i == filter_ques_type for i in x["question_type"]],
                batched=True,
            )

        if max_train_size > 0 and len(self.query_ds) > max_train_size:
            random_select_idx = np.random.randint(
                0, len(self.query_ds), size=max_train_size
            )
            self.query_ds = self.query_ds.select(random_select_idx)

        self.few_shot_num = few_shot_num
        self.instruction = instruction

    def __len__(self):
        return len(self.query_ds)

    def __getitem__(self, index):
        query_item = self.query_ds[index]
        few_shot_index = np.random.randint(0, len(self.ds), size=self.few_shot_num)
        while query_item["idx"] in few_shot_index:
            few_shot_index = np.random.randint(0, len(self.ds), size=self.few_shot_num)
        few_shot_index = few_shot_index.tolist()
        in_context_example = [self.ds[idx] for idx in few_shot_index]

        prompt = []
        if self.instruction:
            prompt = [self.instruction]
        for ice in in_context_example:
            prompt.append(ice["image"])
            prompt.append(f"Question:{ice['question']} Short answer:{ice['answer']}\n")

        query_prompt = [
            query_item["image"],
            f"Question:{query_item['question']} Short answer:{query_item['answer']}",
        ]

        query_x = [ice["image"], f"Question:{query_item['question']} Short answer:"]
        return {
            "ice_prompt": prompt,
            "query_prompt": query_prompt,
            "query_x": query_x,
        }

from functools import partial

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from .icv_datasets.vqa_dataset import VQAV2Dataset


class VQAICVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_cfg,
        processor,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if data_cfg.model_path is None:
            pass
        else:
            self.VLM_processor = CLIPProcessor.from_pretrained(data_cfg.model_path)
        self.processor = processor

        self.collator_data = partial(
            collator_data, processor=self.processor, VLM_processor=self.VLM_processor
        )
        self.data_cfg = data_cfg

    def setup(self, stage: str) -> None:
        # if stage == "fit":
        self.train_ds = VQAV2Dataset(
            name=self.data_cfg.dataset.name,
            root_dir=self.data_cfg.dataset.root_dir,
            train_coco_dataset_root=self.data_cfg.dataset.train_coco_dataset_root,
            val_coco_dataset_root=self.data_cfg.dataset.val_coco_dataset_root,
            instruction=self.data_cfg.dataset.instruction,
            few_shot_num=self.data_cfg.dataset.few_shot_num,
            max_train_size=self.data_cfg.dataset.max_train_size,
            split="train",
            val_ann_file=getattr(self.data_cfg.dataset, "val_ann_file", None),
            # filter_ques_type=self.data_cfg.dataset.filter_ques_type,
            retriver=self.data_cfg.dataset.retriver,
            faiss_directory=self.data_cfg.dataset.faiss_directory,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            self.data_cfg.bs,
            num_workers=self.data_cfg.num_workers,
            collate_fn=self.collator_data,
        )


def collator_data(data_list, processor, VLM_processor):
    sample = data_list[0]
    data_dict = {k: [d[k] for d in data_list] for k in sample.keys()}
    query_prompt = data_dict["query_prompt"]
    ice_prompt = data_dict["ice_prompt"]
    query_x = data_dict["query_x"]

    query_input = processor(
        query_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_eos_token=True,
    )
    if isinstance(VLM_processor, CLIPProcessor):
        texts, images = VLM_process_data(ice_prompt)
        clip_input = []  # the length of it is batchsize
        for i in range(len(texts)):
            clip_batch_input = VLM_processor(
                text=texts[i],
                images=images[i],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            clip_input.append(clip_batch_input)
    else:
        clip_input = None
    query_x = processor(query_x, return_tensors="pt", padding=True, truncation=True)
    ice_input = processor(
        ice_prompt, return_tensors="pt", padding=True, truncation=True
    )
    input_prompts = []

    for ice, query in zip(ice_prompt, query_prompt):
        input_prompts.append(ice + query)

    inputs = processor(
        input_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_eos_token=True,
    )
    in_context_length = (
        ice_input["input_ids"] != processor.tokenizer.pad_token_id
    ).sum(dim=1)
    query_x_length = (query_x["input_ids"] != processor.tokenizer.pad_token_id).sum(
        dim=1
    )
    return {
        "query_input": query_input,
        "inputs": inputs,
        "in_context_length": in_context_length,
        "query_x_length": query_x_length,
        "clip_input": clip_input,
    }


def VLM_process_data(query_prompt):
    batch_texts = []
    batch_images = []
    for sublist in query_prompt:
        texts = []
        images = []
        for item in sublist:
            if isinstance(item, str):
                texts.append(item)
            else:
                images.append(item)
        batch_texts.append(texts)
        batch_images.append(images)
    return batch_texts, batch_images

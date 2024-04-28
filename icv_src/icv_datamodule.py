from functools import partial

import hydra
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoProcessor, IdeficsProcessor


class VQAICVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_cfg,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lvlm_processor = IdeficsProcessor.from_pretrained(data_cfg.model_name)
        self.lvlm_processor.tokenizer.padding_side = "right"

        self.collator_data = partial(collator_data, processor=self.lvlm_processor)
        self.ds_factory = hydra.utils.instantiate(
            self.hparams.data_cfg.dataset, _partial_=True
        )
        self.data_cfg = data_cfg

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_ds = self.ds_factory()

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            self.data_cfg.bs,
            num_workers=self.data_cfg.num_workers,
            collate_fn=self.collator_data,
        )


def collator_data(data_list, processor):
    sample = data_list[0]
    data_dict = {k: [d[k] for d in data_list] for k in sample.keys()}
    query_prompt = data_dict["query_prompt"]
    ice_prompt = data_dict["ice_prompt"]
    query_x = data_dict["query_x"]

    query_input = processor(
        query_prompt, return_tensors="pt", padding=True, truncation=True
    )
    query_x = processor(query_x, return_tensors="pt", padding=True, truncation=True)
    ice_input = processor(
        ice_prompt, return_tensors="pt", padding=True, truncation=True
    )
    input_prompts = []

    for ice, query in zip(ice_prompt, query_prompt):
        input_prompts.append(ice + query)

    inputs = processor(
        input_prompts, return_tensors="pt", padding=True, truncation=True
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
    }

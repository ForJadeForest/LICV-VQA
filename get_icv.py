import datetime
import json
import os
import random
import uuid
from pathlib import Path

import hydra
import pandas as pd
import torch
from dotenv import load_dotenv
import more_itertools
from omegaconf import DictConfig
from PIL import Image

from tqdm import tqdm
from transformers import IdeficsProcessor

from icv_src.icv_datasets.vqa_dataset import load_vqav2_ds, load_okvqa_ds
from icv_src.icv_model.icv_idefics import ICVIdeficsForVisionText2Text
from icv_src.metrics.vqa_metric import (
    compute_vqa_accuracy,
    postprocess_vqa_generation,
)


@hydra.main(config_path="config", config_name="get_icv.yaml")
def main(cfg: DictConfig):
    icv_length = cfg.icv_length
    alpha_value = cfg.alpha
    if cfg.data_cfg.dataset.name == "vqav2":
        train_ds = load_vqav2_ds(
            cfg.data_cfg.dataset.root_dir,
            cfg.data_cfg.dataset.train_coco_dataset_root,
            cfg.data_cfg.dataset.val_coco_dataset_root,
            "train",
        )
    else:
        train_ds = load_okvqa_ds(
            cfg.data_cfg.dataset.root_idr,
            cfg.data_cfg.dataset.train_coco_dataset_root,
            cfg.data_cfg.dataset.val_coco_dataset_root,
            "train",
        )
    if not os.path.exists(cfg.result_path):
        os.makedirs(cfg.result_path)
    model = ICVIdeficsForVisionText2Text.from_pretrained("HuggingFaceM4/idefics-9b")
    model = model.to(cfg.device, torch.bfloat16)
    processor = IdeficsProcessor.from_pretrained("HuggingFaceM4/idefics-9b")
    instruction = cfg.data_cfg.dataset.instrction
    icv_length = cfg.icv_length
    prompts = []
    prompts.extend([instruction])
    for batch in more_itertools.chunked(
        tqdm(train_ds, total=len(train_ds)), icv_length
    ):
        for i, sample in enumerate(batch):
            prompts.extend(
                [
                    sample["image"],
                    f"Question:{sample['question']} Short answer:{sample['answers'][0]['answer']}",
                ]
            )
        break

    alpha = torch.full(size=(1, 33), fill_value=alpha_value).to(cfg.device)
    query_inputs = processor(prompts)
    query_inputs = {k: v.to(model.device) for k, v in query_inputs.items()}
    with torch.no_grad():
        output = model(**query_inputs, output_hidden_states=True, return_dict=True)
        icv = [hidden_state[:, -1] for hidden_state in output.hidden_states]
        icv = torch.stack(icv, dim=1)
        icv = icv[:, 1:, :]
    result_dict = {}

    result_dict["icv_encoder.icv"] = icv
    result_dict["icv_encoder.alpha"] = alpha

    result_path = (
        Path(cfg.result_path) / "icv_length"
        + str(icv_length)
        + "alpha:"
        + str(alpha_value)
    )
    result_path = Path(cfg.result_path) / Path(
        "icv_length" + str(icv_length) + "alpha:" + str(alpha_value)
    )
    os.makedirs(result_path, exist_ok=True)
    dict_path = Path(result_path) / "icv_cpk.bin"
    print("ICV saved to", dict_path)
    torch.save(result_dict, dict_path)


if __name__ == "__main__":
    load_dotenv()
    main()

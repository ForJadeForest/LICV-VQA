import datetime
import json
import os
import random
import shutil
import uuid
from pathlib import Path

import hydra
import more_itertools
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from loguru import logger
from torch import nn
from tqdm import tqdm
from transformers import IdeficsProcessor

from icv_src.icv_datasets.vqa_dataset import load_okvqa_ds, load_vqav2_ds
from icv_src.icv_model.icv_idefics import ICVIdeficsForVisionText2Text
from icv_src.metrics import (
    compute_vqa_accuracy,
    postprocess_ok_vqa_generation,
    postprocess_vqa_generation,
)
from lora_train import FTVQAModule


@hydra.main(config_path="config", config_name="lora_inference.yaml")
def main(cfg: DictConfig):
    from peft import LoraConfig, get_peft_model

    result_dir = Path(cfg.result_dir)
    model_name = cfg.model_name_or_path.split("/")[-1]
    load_model_cpk = (
        result_dir / "model_cpk" / cfg.data_cfg.dataset.name / model_name / cfg.run_name
    )
    save_path: Path = (
        result_dir / "inference" / model_name / cfg.data_cfg.dataset.name / cfg.run_name
    )
    if (save_path / "result.json").exists():
        logger.info("Metrics File exsits, EXIT...")
        return

    if not save_path.exists():
        save_path.mkdir(parents=True)
    meta_info_dir = save_path / "meta_info"
    if not meta_info_dir.exists():
        meta_info_dir.mkdir()
    base_info = f"{str(datetime.datetime.now())}-{cfg.test_num=}-"
    cpk_path = load_model_cpk / "last.ckpt"

    output_file = load_model_cpk / "lightning_module.bin"
    if not output_file.exists():
        convert_zero_checkpoint_to_fp32_state_dict(cpk_path, output_file)

    model = ICVIdeficsForVisionText2Text.from_pretrained(cfg.model_name_or_path)
    processor = IdeficsProcessor.from_pretrained(cfg.model_name_or_path)

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
    module = FTVQAModule.load_from_checkpoint(
        output_file,
        model=model,
        processor=processor,
        map_location="cpu",
    )

    os.remove(output_file)

    model = module.model
    model = model.to("cuda", torch.bfloat16)
    if cfg.data_cfg.dataset.name == "vqav2":
        ds = load_vqav2_ds(
            cfg.data_cfg.dataset.root_dir,
            cfg.data_cfg.dataset.train_coco_dataset_root,
            cfg.data_cfg.dataset.val_coco_dataset_root,
            "validation",
            val_ann_file=cfg.data_cfg.dataset.val_ann_path,
        )
        post_process_fun = postprocess_vqa_generation
    elif cfg.data_cfg.dataset.name == "okvqa":
        ds = load_okvqa_ds(
            cfg.data_cfg.dataset.root_dir,
            cfg.data_cfg.dataset.train_coco_dataset_root,
            cfg.data_cfg.dataset.val_coco_dataset_root,
            "validation",
        )
        post_process_fun = postprocess_ok_vqa_generation
    else:
        raise ValueError(f"{cfg.data_cfg.dataset.name=} error")

    val_ds = ds

    if cfg.test_num != -1:
        val_ds = val_ds.select(range(cfg.test_num))

    result_file = save_path / "result.json"
    result_dict = {}
    if result_file.exists():
        result_dict = json.load(open(save_path / "result.json"))

    results_dict = inference(
        val_ds,
        model,
        processor,
        cfg.bs,
        cfg.data_cfg.dataset.instruction,
    )
    preds = []
    for idx in results_dict:
        preds.append(
            {
                "answer": post_process_fun(results_dict[idx]["prediction"])
                .replace("\n", "")
                .strip(),
                "question_id": results_dict[idx]["question_id"],
            }
        )

    val_ques_path = cfg.data_cfg.dataset.val_ques_path
    val_ann_path = cfg.data_cfg.dataset.val_ann_path
    acc = compute_vqa_accuracy(preds, val_ques_path, val_ann_path)
    result_dict[base_info + f"result"] = acc
    logger.info(f"{cfg.run_name} ACC: {acc['overall']}")
    with open(save_path / "result.json", "w") as f:
        json.dump(result_dict, f, indent=4)
    with open(save_path / "meta_info" / f"lora_inference.json", "w") as f:
        json.dump(results_dict, f, indent=4)


@torch.inference_mode()
def inference(
    val_ds,
    model,
    processor,
    bs,
    instruction="",
):
    results_dict = {}

    index = 0

    for batch in more_itertools.chunked(tqdm(val_ds, total=len(val_ds)), bs):
        if instruction:
            prompts = [[instruction] for _ in range(bs)]
        else:
            prompts = [[] for _ in range(bs)]

        for i, sample in enumerate(batch):
            prompts[i].extend(
                [
                    sample["image"],
                    f"Question:{sample['question']} Short answer:",
                ]
            )

        inputs = processor(prompts)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_out = model.generate(
                **inputs,
                max_new_tokens=5,
                num_beams=3,
                length_penalty=0.0,
                min_new_tokens=0,
            )
        prompt_len = int(inputs["attention_mask"].shape[1])
        outputs = generated_out.tolist()

        generated = processor.tokenizer.batch_decode(
            [output[prompt_len:] for output in outputs],
            skip_special_tokens=True,
        )

        for i in range(len(batch)):
            results_dict[index] = {
                "image_id": batch[i]["image_id"],
                "prediction": generated[i],
                "question_id": batch[i]["question_id"],
                "answer_type": batch[i]["answer_type"],
                "answer": batch[i]["answer"],
            }
            index += 1

    return results_dict


if __name__ == "__main__":
    load_dotenv()
    torch.set_grad_enabled(False)
    main()

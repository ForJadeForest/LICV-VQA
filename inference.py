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
from icv_src.metrics import (
    compute_vqa_accuracy,
    postprocess_vqa_generation,
    postprocess_ok_vqa_generation,
)


@hydra.main(config_path="config", config_name="inference.yaml")
def main(cfg: DictConfig):
    result_dir = Path(cfg.result_dir)
    save_path: Path = result_dir / "inference" / cfg.run_name
    if not save_path.exists():
        save_path.mkdir(parents=True)

    if cfg.test_icv:
        model_cpk_dir = result_dir / "model_cpk" / cfg.run_name
        icv_cpk = torch.load(model_cpk_dir / "icv_cpk.bin")
        icv = icv_cpk["icv_encoder.icv"].to(cfg.device)
        alpha = icv_cpk["icv_encoder.alpha"].to(cfg.device)
        if icv_cpk.get("use_sigmoid", None):
            alpha = torch.sigmoid(alpha)
        print("ICV loaded")
    else:
        icv = None
        alpha = None

    if cfg.data_cfg.dataset.name == "vqav2":
        val_ds = load_vqav2_ds(
            cfg.data_cfg.dataset.root_dir,
            cfg.data_cfg.dataset.train_coco_dataset_root,
            cfg.data_cfg.dataset.val_coco_dataset_root,
            "validation",
        )
        post_process_fun = postprocess_vqa_generation
    else:
        val_ds = load_okvqa_ds(
            cfg.data_cfg.dataset.root_dir,
            cfg.data_cfg.dataset.train_coco_dataset_root,
            cfg.data_cfg.dataset.val_coco_dataset_root,
            "validation",
        )
        post_process_fun = postprocess_ok_vqa_generation
    model = ICVIdeficsForVisionText2Text.from_pretrained(cfg.model_name_or_path)
    model = model.to(cfg.device, torch.bfloat16)
    processor = IdeficsProcessor.from_pretrained(cfg.model_name_or_path)
    if cfg.test_num != -1:
        val_ds = val_ds.select(range(cfg.test_num))

    results_dict = inference(
        val_ds,
        model,
        processor,
        cfg.bs,
        cfg.data_cfg.dataset.instruction,
        icv,
        alpha,
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

    with open(save_path / "result.json", "w") as f:
        json.dump(acc, f, indent=4)

    print(acc)


def inference(
    val_ds,
    model,
    processor,
    bs,
    instruction="",
    in_context_vector=None,
    alpha=None,
):
    results_dict = {}

    index = 0

    for batch in more_itertools.chunked(tqdm(val_ds, total=len(val_ds)), bs):
        prompts = []
        if instruction:
            prompts = [[instruction] for _ in range(bs)]
        for i, sample in enumerate(batch):
            prompts[i].extend(
                [
                    sample["image"],
                    f"Question:{sample['question']} Short answer:",
                ]
            )

        query_inputs = processor(prompts)
        query_inputs = {k: v.to(model.device) for k, v in query_inputs.items()}
        with torch.no_grad():
            generated_out = model.generate(
                **query_inputs,
                in_context_vector=in_context_vector,
                alpha=alpha,
                max_new_tokens=5,
                num_beams=3,
                length_penalty=0.0,
                min_new_tokens=0,
            )
        prompt_len = int(query_inputs["attention_mask"].shape[1])
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
            }
            index += 1

    return results_dict


if __name__ == "__main__":
    load_dotenv()
    main()

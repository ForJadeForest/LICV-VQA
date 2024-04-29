import datetime
import json
import os
import random
import uuid
from pathlib import Path
import numpy as np
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
    meta_info_dir = save_path / "meta_info"
    if not meta_info_dir.exists():
        meta_info_dir.mkdir()

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
    split = "validation"
    base_info = f"{str(datetime.datetime.now())}-{cfg.test_num=}-"
    if cfg.test_icl:
        split = None

    if cfg.data_cfg.dataset.name == "vqav2":
        ds = load_vqav2_ds(
            cfg.data_cfg.dataset.root_dir,
            cfg.data_cfg.dataset.train_coco_dataset_root,
            cfg.data_cfg.dataset.val_coco_dataset_root,
            split,
            val_ann_file=cfg.data_cfg.dataset.val_ann_path,
        )

        post_process_fun = postprocess_vqa_generation
    elif cfg.data_cfg.dataset.name == "okvqa":
        ds = load_okvqa_ds(
            cfg.data_cfg.dataset.root_dir,
            cfg.data_cfg.dataset.train_coco_dataset_root,
            cfg.data_cfg.dataset.val_coco_dataset_root,
            split,
        )
        post_process_fun = postprocess_ok_vqa_generation
    else:
        raise ValueError(f"{cfg.data_cfg.dataset.name=} error")
    if cfg.test_icl:
        val_ds = ds["validation"]
        train_ds = ds["train"]
    model = ICVIdeficsForVisionText2Text.from_pretrained(cfg.model_name_or_path)
    model = model.to(cfg.device, torch.bfloat16)
    processor = IdeficsProcessor.from_pretrained(cfg.model_name_or_path)
    if cfg.test_num != -1:
        val_ds = val_ds.select(range(cfg.test_num))

    result_file = save_path / "result.json"
    result_dict = {}
    if result_file.exists():
        result_dict = json.load(open(save_path / "result.json"))

    if cfg.test_icv:
        results_dict = icv_inference(
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

        result_dict[base_info + "icv result"] = acc
        with open(save_path / "result.json", "w") as f:
            json.dump(result_dict, f, indent=4)
        with open(save_path / "meta_info" / f"icv.json", "w") as f:
            json.dump(results_dict, f, indent=4)

    if cfg.test_icl:
        for shot_num in cfg.few_shot_list:

            results_dict = icl_inference(
                train_ds,
                val_ds,
                shot_num,
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
            result_dict[base_info + f"shot{shot_num} result"] = acc

            with open(save_path / "result.json", "w") as f:
                json.dump(result_dict, f, indent=4)
            with open(save_path / "meta_info" / f"icl_shot{shot_num}.json", "w") as f:
                json.dump(results_dict, f, indent=4)


@torch.inference_mode()
def icv_inference(
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


@torch.inference_mode()
def icl_inference(
    train_ds,
    val_ds,
    few_shot_num,
    model,
    processor,
    bs,
    instruction="",
):
    results_dict = {}

    index = 0
    ice_idx_list = []
    ice_idx_sample_list = list(range(len(train_ds)))
    for i in range(len(val_ds)):
        ice_idx = random.sample(ice_idx_sample_list, few_shot_num)
        ice_idx_list.append(ice_idx)

    for batch in more_itertools.chunked(tqdm(val_ds, total=len(val_ds)), bs):
        if instruction:
            prompts = [[instruction] for _ in range(bs)]
        else:
            prompts = [[] for _ in range(bs)]

        sub_ice_idx_list = ice_idx_list[index : index + bs]

        for i, sample in enumerate(batch):
            for ice_idx in sub_ice_idx_list[i]:
                ice = train_ds[ice_idx]
                prompts[i].extend(
                    [
                        ice["image"],
                        f"Question:{ice['question']} Short answer:{ice['answer']}",
                    ]
                )
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

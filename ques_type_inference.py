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


@hydra.main(config_path="config", config_name="ques_type_inference.yaml")
def main(cfg: DictConfig):
    result_dir = Path(cfg.result_dir)
    save_path: Path = (
        result_dir / "inference" / cfg.data_cfg.dataset.name / cfg.run_name
    )
    if not save_path.exists():
        save_path.mkdir(parents=True)
    meta_info_dir = save_path / "meta_info"
    if not meta_info_dir.exists():
        meta_info_dir.mkdir()

    base_info = f"{str(datetime.datetime.now())}-{cfg.test_num=}-"

    if cfg.data_cfg.dataset.name == "vqav2":
        val_ds = load_vqav2_ds(
            cfg.data_cfg.dataset.root_dir,
            cfg.data_cfg.dataset.train_coco_dataset_root,
            cfg.data_cfg.dataset.val_coco_dataset_root,
            "validation",
            val_ann_file=cfg.data_cfg.dataset.val_ann_path,
        )
        post_process_fun = postprocess_vqa_generation
    elif cfg.data_cfg.dataset.name == "okvqa":
        val_ds = load_okvqa_ds(
            cfg.data_cfg.dataset.root_dir,
            cfg.data_cfg.dataset.train_coco_dataset_root,
            cfg.data_cfg.dataset.val_coco_dataset_root,
            "validation",
        )
        post_process_fun = postprocess_ok_vqa_generation
    else:
        raise ValueError(f"{cfg.data_cfg.dataset.name=} error")

    ques_type_list = list(set(val_ds["gen_question_type"]))

    ques_type_icv_dict = {}
    for ques_type in ques_type_list:
        model_name = cfg.model_name_or_path.split("/")[-1]
        model_cpk_dir = (
            result_dir
            / "model_cpk"
            / cfg.data_cfg.dataset.name
            / model_name
            / (f"ques_type:{ques_type}_" + cfg.run_name)
        )
        icv_cpk = torch.load(model_cpk_dir / "icv_cpk.bin")
        icv = icv_cpk["icv_encoder.icv"].to(cfg.device)
        alpha = icv_cpk["icv_encoder.alpha"].to(cfg.device)
        print(icv_cpk.get("use_sigmoid", None), alpha)
        if icv_cpk.get("use_sigmoid", None):
            alpha = torch.sigmoid(alpha)
        ques_type_icv_dict[ques_type] = {"icv": icv, "alpha": alpha}

    model = ICVIdeficsForVisionText2Text.from_pretrained(cfg.model_name_or_path)
    model = model.to(cfg.device, torch.bfloat16)
    processor = IdeficsProcessor.from_pretrained(cfg.model_name_or_path)
    if cfg.test_num != -1:
        val_ds = val_ds.select(range(cfg.test_num))

    result_file = save_path / "result.json"
    result_dict = {}
    if result_file.exists():
        result_dict = json.load(open(save_path / "result.json"))

    results_dict = icv_set_inference(
        val_ds,
        model,
        processor,
        cfg.bs,
        cfg.data_cfg.dataset.instruction,
        ques_type_icv_dict,
        cfg.test_all_mean,
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
    with open(save_path / "query_type_result.json", "w") as f:
        json.dump(result_dict, f, indent=4)
    with open(save_path / "meta_info" / f"icv.json", "w") as f:
        json.dump(results_dict, f, indent=4)


@torch.inference_mode()
def icv_set_inference(
    val_ds, model, processor, bs, instruction="", icv_dict=None, test_all_mean=False
):
    results_dict = {}

    index = 0
    all_mean_icv = []
    all_mean_alpha = []
    for key in icv_dict:
        all_mean_icv.append(icv_dict[key]["icv"])
        all_mean_alpha.append(icv_dict[key]["alpha"])
    all_mean_icv = torch.stack(all_mean_icv).mean(dim=0)
    all_mean_alpha = torch.stack(all_mean_alpha).mean(dim=0)

    for batch in more_itertools.chunked(tqdm(val_ds, total=len(val_ds)), bs):

        if instruction:
            prompts = [[instruction] for _ in range(bs)]
        else:
            prompts = [[] for _ in range(bs)]
        icvs = []
        alphas = []
        if not test_all_mean:
            for i, sample in enumerate(batch):
                prompts[i].extend(
                    [
                        sample["image"],
                        f"Question:{sample['question']} Short answer:",
                    ]
                )
                icvs.extend(
                    [icv_dict[sample["gen_question_type"]]["icv"] for _ in range(3)]
                )
                alphas.extend(
                    [icv_dict[sample["gen_question_type"]]["alpha"] for _ in range(3)]
                )

            icvs = torch.cat(icvs, dim=0)
            alphas = torch.cat(alphas, dim=0)
        else:
            icvs = all_mean_icv
            alphas = all_mean_alpha
        query_inputs = processor(prompts)
        query_inputs = {k: v.to(model.device) for k, v in query_inputs.items()}

        with torch.no_grad():
            generated_out = model.generate(
                **query_inputs,
                in_context_vector=icvs,
                alpha=alphas,
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
    torch.set_grad_enabled(False)
    main()

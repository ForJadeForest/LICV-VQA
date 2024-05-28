import datetime
import json
import random
from pathlib import Path

import hydra
import more_itertools
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import IdeficsProcessor

from icv_src.icv_datasets.vqa_dataset import load_okvqa_ds, load_vqav2_ds
from icv_src.icv_model.icv_idefics import ICVIdeficsForVisionText2Text
from icv_src.icv_model.icv_intervention import LearnableICVInterventionLMM
from icv_src.metrics import (
    compute_vqa_accuracy,
    postprocess_ok_vqa_generation,
    postprocess_vqa_generation,
)
from baukit import TraceDict, get_module
from utils import get_icv_cpk_path, get_inference_paths


@hydra.main(config_path="config", config_name="inference.yaml")
def main(cfg: DictConfig):
    logger.info(f"begin run: {cfg.run_name}")
    result_dir = Path(cfg.result_dir)
    model_name = cfg.model_name_or_path.split("/")[-1]

    save_dir, meta_info_dir, metric_file_path = get_inference_paths(
        result_dir=result_dir,
        model_name=model_name,
        dataset_name=cfg.data_cfg.dataset.name,
        run_name=cfg.run_name,
    )

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    if not meta_info_dir.exists():
        meta_info_dir.mkdir()

    if not metric_file_path.exists():
        result_dict = {}
    elif cfg.re_eval:
        result_dict = json.load(open(metric_file_path))
        logger.info(f"{metric_file_path} exists. LOADING...")
    elif not cfg.re_eval:
        logger.info(f"{metric_file_path} exists, EXIT...")
        return

    icv = None
    alpha = None
    if cfg.test_icv:
        model_cpk_dir = get_icv_cpk_path(
            result_dir,
            model_name=model_name,
            dataset_name=cfg.data_cfg.dataset.name,
            run_name=cfg.run_name,
        )
        icv_cpk = torch.load(model_cpk_dir / "icv_cpk.pth")
        icv = icv_cpk["icv_encoder.icv"].to(cfg.device)
        alpha = icv_cpk["icv_encoder.alpha"].to(cfg.device)
        intervention_lmm_args = dict(icv_cpk["icv_intervention_args"])
        if icv_cpk.get("use_sigmoid", None):
            alpha = torch.sigmoid(alpha)
        logger.info("ICV loaded")

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
    else:
        val_ds = ds
    model = ICVIdeficsForVisionText2Text.from_pretrained(cfg.model_name_or_path)
    model = model.to(cfg.device, torch.bfloat16)
    intervention_lmm_args.pop("model_name")
    icv_model = LearnableICVInterventionLMM(model, **intervention_lmm_args)
    processor = IdeficsProcessor.from_pretrained(cfg.model_name_or_path)
    if cfg.test_num != -1:
        val_ds = val_ds.select(range(cfg.test_num))

    if cfg.test_icv:
        icv_model.toggle_intervention(True)
        logger.info(f"{icv_model.intervention_enabled=}")
        results_dict = icv_inference(
            val_ds=val_ds,
            model=icv_model,
            processor=processor,
            bs=cfg.bs,
            generate_kwargs=cfg.generate_kwargs,
            instruction=cfg.data_cfg.dataset.instruction,
            in_context_vector=icv,
            alpha=alpha,
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
        logger.info(f"{cfg.run_name} ACC: {acc['overall']}")
        result_dict[base_info + "icv result"] = acc
        with open(metric_file_path, "w") as f:
            json.dump(result_dict, f, indent=4)
        with open(meta_info_dir / f"{base_info}icv.json", "w") as f:
            json.dump(results_dict, f, indent=4)

    if cfg.test_icl:
        icv_model.toggle_intervention(False)
        for shot_num in cfg.few_shot_list:

            results_dict = icl_inference(
                train_ds=train_ds,
                val_ds=val_ds,
                shot_num=shot_num,
                model=icv_model,
                processor=processor,
                bs=cfg.bs,
                generate_kwargs=cfg.generate_kwargs,
                instruction=cfg.data_cfg.dataset.instruction,
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

            with open(metric_file_path, "w") as f:
                json.dump(result_dict, f, indent=4)
            with open(meta_info_dir / f"icl_shot{shot_num}.json", "w") as f:
                json.dump(results_dict, f, indent=4)


@torch.inference_mode()
def icv_inference(
    val_ds,
    model,
    processor,
    bs,
    generate_kwargs,
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
        query_inputs = {k: v.to(model.lmm.device) for k, v in query_inputs.items()}

        generated = generate_answers(
            inputs=query_inputs,
            model=model,
            processor=processor,
            generate_kwargs=generate_kwargs,
            in_context_vector=in_context_vector,
            alpha=alpha,
        )

        for i in range(len(batch)):
            batch[i].pop("image")
            results_dict[index] = {
                "prediction": generated[i],
                **batch[i],
            }
            index += 1

    return results_dict


@torch.inference_mode()
def generate_answers(
    inputs,
    model,
    processor,
    generate_kwargs,
    in_context_vector=None,
    alpha=None,
):
    icv = None
    if in_context_vector is not None:
        icv = alpha.unsqueeze(dim=-1) * in_context_vector

    generated_out = model.generate(inputs, generate_kwargs, icv=icv)
    prompt_len = int(inputs["attention_mask"].shape[1])
    outputs = generated_out.tolist()

    generated = processor.tokenizer.batch_decode(
        [output[prompt_len:] for output in outputs],
        skip_special_tokens=True,
    )
    return generated


@torch.inference_mode()
def icl_inference(
    train_ds,
    val_ds,
    few_shot_num,
    model,
    processor,
    bs,
    generate_kwargs,
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
        generated = generate_answers(
            inputs=inputs,
            model=model,
            processor=processor,
            generate_kwargs=generate_kwargs,
        )

        for i in range(len(batch)):
            results_dict[index] = {"prediction": generated[i], **batch[i]}
            index += 1

    return results_dict


if __name__ == "__main__":
    load_dotenv()
    torch.set_grad_enabled(False)
    main()

from pathlib import Path

from icv_src.icv_datasets.vqa_dataset import load_okvqa_ds, load_vqav2_ds
from icv_src.icv_datasets.caption_dataset import load_coco_ds
from icv_src.metrics import (
    postprocess_ok_vqa_generation,
    postprocess_vqa_generation,
)
from lmm_icl_interface import (
    Idefics2Interface,
    IdeficsInterface,
    LMMPromptManager,
    OpenFlamingoInterface,
)


def get_icv_cpk_path(result_dir, model_name, dataset_name, run_name):
    result_dir = Path(result_dir)
    model_cpk_dir = result_dir / "model_cpk" / dataset_name / model_name / run_name
    return model_cpk_dir


def get_inference_paths(result_dir, model_name, dataset_name, run_name):
    result_dir = Path(result_dir)
    save_dir = result_dir / "inference" / model_name / dataset_name / run_name
    meta_info_dir = save_dir / "meta_info"
    metric_file_path = save_dir / "result.json"
    return save_dir, meta_info_dir, metric_file_path


def init_interface(cfg):
    model_cpk_dir = Path(cfg.model_cpk_dir)
    prompt_manager = LMMPromptManager(
        cfg.prompt.prompt_template,
        cfg.prompt.column_token_map,
        label_field=cfg.prompt.label_filed,
        sep_token=cfg.prompt.sep_token,
        query_prompt_template=cfg.prompt.query_prompt_template,
    )
    if cfg.lmm.name == "idefics-9b":
        interface = IdeficsInterface(
            model_name_or_path=model_cpk_dir / cfg.lmm.model_name,
            precision=cfg.lmm.precision,
            device=cfg.lmm.device,
            prompt_manager=prompt_manager,
            instruction=cfg.prompt.instruction,
            image_field=cfg.prompt.image_field,
            label_field=cfg.prompt.label_filed,
        )
        processor = interface.processor
    elif "openflamingo" in cfg.lmm.name.lower():
        interface = OpenFlamingoInterface(
            lang_encoder_path=model_cpk_dir / cfg.lmm.lang_encoder_path,
            tokenizer_path=model_cpk_dir / cfg.lmm.tokenizer_path,
            flamingo_checkpoint_dir=model_cpk_dir / cfg.lmm.hf_root,
            cross_attn_every_n_layers=cfg.lmm.cross_attn_every_n_layers,
            hf_root=cfg.lmm.hf_root,
            precision=cfg.lmm.precision,
            device=cfg.lmm.device,
            prompt_manager=prompt_manager,
            instruction=cfg.prompt.instruction,
            image_field=cfg.prompt.image_field,
            label_field=cfg.prompt.label_filed,
            load_from_local=True,
            init_device=cfg.lmm.init_device,
        )
        processor = interface.processor
    elif cfg.lmm.name == "idefics2-8b-base":
        interface = Idefics2Interface(
            model_name_or_path=model_cpk_dir / cfg.lmm.model_name,
            precision=cfg.lmm.precision,
            device=cfg.lmm.device,
            prompt_manager=prompt_manager,
            instruction=cfg.prompt.instruction,
            image_field=cfg.prompt.image_field,
            label_field=cfg.prompt.label_filed,
        )
        processor = interface.processor

    return prompt_manager, interface, processor


def init_dataset(cfg, split):
    if cfg.data_cfg.task.datasets.name == "vqav2":
        ds = load_vqav2_ds(
            cfg.data_cfg.task.datasets.root_dir,
            cfg.data_cfg.task.datasets.train_coco_dataset_root,
            cfg.data_cfg.task.datasets.val_coco_dataset_root,
            split,
            val_ann_file=cfg.data_cfg.task.datasets.val_ann_path,
        )
        post_process_fun = vqa_postprocess
    elif cfg.data_cfg.task.datasets.name == "okvqa":
        ds = load_okvqa_ds(
            cfg.data_cfg.task.datasets.root_dir,
            cfg.data_cfg.task.datasets.train_coco_dataset_root,
            cfg.data_cfg.task.datasets.val_coco_dataset_root,
            split,
        )
        post_process_fun = ok_vq_postprocess
    elif cfg.data_cfg.task.datasets.name == "coco2017":
        ds = load_coco_ds(
            train_coco_dataset_root=cfg.data_cfg.task.datasets.train_coco_dataset_root,
            val_coco_dataset_root=cfg.data_cfg.task.datasets.val_coco_dataset_root,
            train_coco_annotation_file=cfg.data_cfg.task.datasets.train_coco_annotation_file,
            val_coco_annotation_file=cfg.data_cfg.task.datasets.val_coco_annotation_file,
            split=split,
        )
        post_process_fun = caption_postprocess
    else:
        raise ValueError(f"{cfg.data_cfg.task.datasets.name=} error")
    return ds, post_process_fun


def caption_postprocess(text, model_name):
    if "flamingo" in model_name:
        return text.split("Output", 1)[0].replace('"', "").strip()
    elif "idefics" in model_name:
        return text.split("Caption", 1)[0].replace('"', "").replace("\n", "").strip()


def vqa_postprocess(text, model_name):
    if "flamingo" in model_name:
        return postprocess_vqa_generation(text).strip()
    elif "idefics" in model_name:
        return postprocess_vqa_generation(text).replace("\n", "").strip()


def ok_vq_postprocess(text, model_name):
    if "flamingo" in model_name:
        return postprocess_ok_vqa_generation(text).strip()
    elif "idefics" in model_name:
        return postprocess_ok_vqa_generation(text).replace("\n", "").strip()

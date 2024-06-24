from pathlib import Path

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

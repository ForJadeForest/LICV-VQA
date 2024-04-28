import os
from pathlib import Path

import hydra
import more_itertools
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import IdeficsProcessor

from icv_src.icv_datasets.vqa_dataset import load_okvqa_ds, load_vqav2_ds
from icv_src.icv_model.icv_idefics import ICVIdeficsForVisionText2Text


@hydra.main(config_path="config", config_name="heuristic_icv.yaml")
def main(cfg: DictConfig):
    result_path = Path(cfg.result_path)

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
    if not result_path.exists():
        result_path.mkdir()

    model_cpk_path = Path(cfg.model_cpk_dir) / cfg.mdoel_name
    model = ICVIdeficsForVisionText2Text.from_pretrained(model_cpk_path)
    model = model.to(cfg.device, torch.bfloat16)
    processor = IdeficsProcessor.from_pretrained(model_cpk_path)

    prompts = []
    prompts.extend([cfg.data_cfg.dataset.instruction])

    ice_list = [train_ds[i] for i in range(cfg.icv_length)]

    for sample in ice_list:
        prompts.extend(
            [
                sample["image"],
                f"Question:{sample['question']} Short answer:{sample['answers'][0]['answer']}",
            ]
        )

    alpha = torch.full(size=(1, 33), fill_value=cfg.alpha_value).to(cfg.device)
    ice_inputs = processor(prompts)
    ice_inputs = {k: v.to(model.device) for k, v in ice_inputs.items()}
    with torch.no_grad():
        output = model(**ice_inputs, output_hidden_states=True, return_dict=True)
        icv = [hidden_state[:, -1] for hidden_state in output.hidden_states]
        icv = torch.stack(icv, dim=1)
        icv = icv[:, 1:, :]
    result_dict = {}

    result_dict["icv_encoder.icv"] = icv
    result_dict["icv_encoder.alpha"] = alpha

    result_path = result_path / f"icv_length:{cfg.icv_length}-alpha:{cfg.alpha_value}"
    result_path.mkdir(parents=True)
    dict_path = result_path / "icv_cpk.bin"
    print("ICV saved to", dict_path)
    torch.save(result_dict, dict_path)


if __name__ == "__main__":
    load_dotenv()
    main()

import os
from pathlib import Path

import hydra
import more_itertools
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import IdeficsProcessor
from icv_src.utils.pca import PCA

from icv_src.icv_datasets.vqa_dataset import load_okvqa_ds, load_vqav2_ds
from icv_src.icv_model.icv_idefics import ICVIdeficsForVisionText2Text
from random import sample
import random


@hydra.main(config_path="config", config_name="heuristic_icv.yaml")
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    def get_icv(input):
        ice_inputs = processor(input)
        ice_inputs = {k: v.to(model.device) for k, v in ice_inputs.items()}
        with torch.no_grad():
            output = model(**ice_inputs, output_hidden_states=True, return_dict=True)
            icv = [hidden_state[:, -1] for hidden_state in output.hidden_states]
            icv = torch.stack(icv, dim=1)
            icv = icv[:, 1:, :]
        return icv

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

    # model_cpk_path = Path(cfg.model_cpk_dir) / cfg.mdoel_name
    model = ICVIdeficsForVisionText2Text.from_pretrained("/data/share/ideficsbase/")
    model = model.to(cfg.device, torch.bfloat16)
    processor = IdeficsProcessor.from_pretrained("/data/share/ideficsbase/")
    icv_x = []
    icv_y = []
    # select cfg.length samples from the dataset
    alpha = torch.full(size=(1, 33), fill_value=cfg.alpha_value).to(cfg.device)
    for i in range(cfg.icv_range):

        prompt_x = []
        prompt_y = []
        prompt_x.extend([cfg.data_cfg.dataset.instruction])
        random_list = sample(range(len(train_ds)), 1)
        for idx in random_list:
            prompt_x.extend(
                [train_ds[idx]["image"], f"Question:{train_ds[idx]['question']}"]
            )
            prompt_y.extend([f" Short answer:{train_ds[idx]['answers'][0]['answer']}"])
        # include answer and list encoding
        icv_x.append(get_icv(prompt_x))
        icv_y.append(get_icv(prompt_y))

    hidden_states_all = []
    num_demonstration = len(icv_x)
    for demonstration_id in range(num_demonstration):
        h = icv_x[demonstration_id].flatten() - icv_y[demonstration_id].flatten()
        hidden_states_all.append(h)
    _, num_layers, hidden_dim = icv_x[0].size()
    fit_data = torch.stack(hidden_states_all)
    pca = PCA(n_components=1).to(fit_data.device).fit(fit_data.float())
    direction = (pca.components_.sum(dim=0, keepdim=True) + pca.mean_).mean(0)
    result_dict = {}

    result_dict["icv_encoder.icv"] = direction.view(1, num_layers, hidden_dim)

    result_dict["icv_encoder.alpha"] = alpha

    result_path = (
        result_path
        / f"model_cpk"
        / f"pca_range:{cfg.icv_range}-alpha:{cfg.alpha_value}"
    )
    result_path.mkdir(parents=True)
    dict_path = result_path / "icv_cpk.bin"
    print("ICV saved to", dict_path)
    torch.save(result_dict, dict_path)


if __name__ == "__main__":
    load_dotenv()
    main()

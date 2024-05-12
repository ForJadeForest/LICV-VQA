import os
import shutil
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from loguru import logger
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from transformers import IdeficsProcessor

from icv_src.icv_datamodule import VQAICVDataModule
from icv_src.icv_model.icv_idefics import ICVIdeficsForVisionText2Text
from icv_src.icv_module import VQAICVModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="config", config_name="train.yaml")
def main(cfg: DictConfig):
    global logger
    pl.seed_everything(cfg.seed)
    if not os.path.exists(cfg.result_dir):
        os.makedirs(cfg.result_dir)

    model_name = cfg.model_name.split("/")[-1]
    save_path = os.path.join(
        cfg.result_dir,
        "model_cpk",
        cfg.data_cfg.dataset.name,
        model_name,
        cfg.run_name,
    )
    save_path = Path(save_path)
    if (save_path / "icv_cpk.bin").exists():
        logger.info(f"{str(save_path / 'icv_cpk.bin')} exists! exit...")
        return
    wb_logger = WandbLogger(
        save_dir=cfg.result_dir,
        name=cfg.run_name,
        project="VQAInContextVector",
        log_model=False,
    )
    wb_logger.log_hyperparams(dict(cfg))
    model_cpk_callback = ModelCheckpoint(
        filename="min_tl-{epoch}-{loss:.5f}",
        monitor="loss",
        save_last=True,
        save_top_k=0,
        mode="min",
        save_weights_only=True,
        dirpath=save_path,
    )
    trainer = pl.Trainer(
        logger=wb_logger,
        callbacks=[
            LearningRateMonitor(),
            RichModelSummary(max_depth=2),
            RichProgressBar(),
            # model_cpk_callback,
        ],
        **cfg.trainer,
        enable_checkpointing=False,
    )
    model_path = Path(cfg.model_cpk_dir) / cfg.model_name

    model = ICVIdeficsForVisionText2Text.from_pretrained(model_path)
    processor = IdeficsProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "right"

    model = VQAICVModule(
        model=model,
        processor=processor,
        module_cfg=cfg.icv_module,
    )
    data_module = VQAICVDataModule(data_cfg=cfg.data_cfg, processor=processor)

    trainer.fit(
        model,
        data_module,
    )
    trainer.save_checkpoint(
        filepath=os.path.join(
            save_path,
            "last.ckpt",
        ),
        weights_only=True,
    )
    postprocess(cfg, save_path)


@rank_zero_only
def postprocess(cfg, save_path):
    save_path = Path(save_path)
    if "deepspeed" in cfg.trainer.strategy:
        cpk_save_path = save_path / "last.ckpt"
        output_file = save_path / "lightning_module.bin"
        convert_zero_checkpoint_to_fp32_state_dict(cpk_save_path, output_file)

        checkpoint = torch.load(output_file)
        params_name = list(checkpoint["state_dict"].keys())
        for name in params_name:
            if name.startswith("model"):
                checkpoint["state_dict"].pop(name)
        checkpoint["state_dict"]["use_sigmoid"] = getattr(
            cfg.icv_module.icv_encoder, "use_sigmoid", None
        )
        torch.save(checkpoint["state_dict"], save_path / "icv_cpk.bin")
        os.remove(output_file)
        shutil.rmtree(
            cpk_save_path,
        )


if __name__ == "__main__":
    load_dotenv()
    main()

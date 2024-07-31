# L-ICV: Learnable In-Context Vector for Visual Question Answering

# Install 
```shell
conda create -n licv python=3.10

conda activate licv
pip install -r requirments.txt

# For Openflamingo, please use transformers==4.28.1 [beta]

pip install transformers==4.28.1
```


## Run
```shell
# Run Idefics-v1-9B on vqav2 with 32 shot
python train.py run_name="vqav2_idefics_icv"\
                icv_module.icv_encoder.use_sigmoid=False\
                icv_module.icv_encoder.alpha_init_value=0.1\
                data_cfg.dataset.max_train_size=8000\
                data_cfg.dataset.few_shot_num=32\
                data_cfg.bs=8\
                data_cfg.num_workers=10\
                trainer.accumulate_grad_batches=2\
                trainer.devices=4 \
                icv_module.icv_lr=1e-3 \
                icv_module.hard_loss_weight=0.5 \
                data_cfg/task/datasets=vqav2 \
                lmm=idefics-9B\
                trainer.precision="16-mixed" 


# Run Idefics-v1-9B on okvqa with 32 shot
python train.py run_name="okvqa_idefics_icv"\
                icv_module.icv_encoder.use_sigmoid=False\
                icv_module.icv_encoder.alpha_init_value=0.1\
                data_cfg.dataset.max_train_size=8000\
                data_cfg.dataset.few_shot_num=32\
                data_cfg.bs=8\
                data_cfg.num_workers=10\
                trainer.accumulate_grad_batches=2\
                trainer.devices=4 \
                icv_module.icv_lr=5e-3 \
                icv_module.hard_loss_weight=0.5 \
                data_cfg/task/datasets=ok_vqa \
                lmm=idefics-9B\
                trainer.precision="16-mixed" 


# Run Idefics-v2-9B on vqav2 with 1 shot
python train.py run_name="vqav2_idefics2_icv"\
                icv_module.icv_encoder.use_sigmoid=False\
                icv_module.icv_encoder.alpha_init_value=0.1\
                data_cfg.dataset.max_train_size=8000\
                data_cfg.dataset.few_shot_num=1\
                data_cfg.bs=8\
                data_cfg.num_workers=10\
                trainer.accumulate_grad_batches=2\
                trainer.devices=4 \
                icv_module.icv_lr=1e-3 \
                icv_module.hard_loss_weight=0.5 \
                data_cfg/task/datasets=vqav2 \
                lmm=idefics2-8B-base\
                trainer.precision="bf16-mixed" 


# Run Idefics-v2-9B on okvqa with 1 shot
python train.py run_name="okvqa_idefics2_icv"\
                icv_module.icv_encoder.use_sigmoid=False\
                icv_module.icv_encoder.alpha_init_value=0.1\
                data_cfg.dataset.max_train_size=8000\
                data_cfg.dataset.few_shot_num=1\
                data_cfg.bs=8\
                data_cfg.num_workers=10\
                trainer.accumulate_grad_batches=2\
                trainer.devices=4 \
                icv_module.icv_lr=5e-3 \
                icv_module.hard_loss_weight=0.5 \
                data_cfg/task/datasets=ok_vqa \
                lmm=idefics2-8B-base\
                trainer.precision="bf16-mixed" 
```

## Run Inference

```shell
python inference.py run_name="vqav2_idefics_icv" \
                data_cfg/task/datasets=vqav2\
                lmm=idefics-9B

python inference.py run_name="okvqa_idefics_icv" \
                data_cfg/task/datasets=ok_vqa\
                lmm=idefics-9B

python inference.py run_name="okvqa_idefics2_icv" \
                data_cfg/task/datasets=ok_vqa\
                lmm=idefics2-8B-base

python inference.py run_name="vqav2_idefics_icv" \
                data_cfg/task/datasets=vqav2\
                lmm=idefics2-8B-base
```
# Training and Evaluation

We provide bash scripts in [scripts/](../scripts) for each prompting variant including LAMM, CoOp+LAMM, MaPLe+LAMM.

Make sure to configure the dataset paths in environment variable `DATA` and run the commands from the main directory.

Below we provide training and evaluation instructions for LAMM. 


### Training 
We train LAMM using a **single** NVIDIA A100 GPU.

## LAMM training

### LAMM setting
The default training settings are provided in config file at `configs/trainers/CoOp/vit_b16_ep50_ctxv1.yaml`. All hyper-parameters such as prompt length, prompt depth, etc., can be modified using this config file.

Below, we provide instructions to train CLIP+LAMM on all datasets and seed 1, 2, 3. 


```bash
bash scripts/lamm/base_train_lamm_all.sh
```

### CoOp+LAMM setting
The default training settings are provided in config file at `configs/trainers/CoOp/vit_b16_ep50_ctxv1.yaml`. All hyper-parameters such as prompt length, prompt depth, etc., can be modified using this config file.

Below, we provide instructions to train CoOp+LAMM on all datasets and seed 1, 2, 3. 


```bash
bash scripts/coop/base_train_coop_lamm_all.sh
```

### MaPLe+LAMM setting
The default training settings are provided in config file at `configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml`. All hyper-parameters such as prompt length, prompt depth, etc., can be modified using this config file.

Below, we provide instructions to train MaPLe+LAMM on all datasets and seed 1, 2, 3. 


```bash
bash scripts/maple/base_train_maple_lamm_all.sh
```

Now use the script `parse_test_res.py` and run the commands below to calculate t
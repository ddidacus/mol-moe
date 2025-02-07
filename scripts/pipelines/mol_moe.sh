#!/bin/bash

# (1)-(2)-(3)
bash scripts/train_experts.sh

# (4) Merge the expert into the base moe model
model_name="models/mol-moe-base-6x1b"
if [ ! -d $model_name ]; then
    mergekit-moe configs/merging_moe.yaml $model_name
fi

# (5) Train the MoE routers model with RL
echo "[-] Training the MoE routers model with RL"
python scripts/rl/rloo.py --config configs/mol-moe.yaml
accelerate launch --config-file configs/accelerate.yaml scripts/rl/rloo.py --config configs/mol-moe.yaml


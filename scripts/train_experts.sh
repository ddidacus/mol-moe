#!/bin/bash

# (1)-(2)
bash scripts/train_base_model.sh

# (3) Train the expert models with RL
tasks=("JNK3" "DRD2" "GSK3B" "CYP2C19_Veith" "CYP2D6_Veith")
for task in "${tasks[@]}"; do
    if [ ! -d "models/mol-llama-1b-$task/consolidated" ]; then
        echo "[-] Training experts $task..."
        accelerate launch --config-file configs/accelerate.yaml scripts/rl/rloo.py --config configs/mol_llama_$task.yaml
    fi
done

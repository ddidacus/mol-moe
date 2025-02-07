#!/bin/bash

# Check ablated flag
ablated_flag=""
if [ "$1" == "--ablated" ]; then
    ablated_flag="--ablated"
fi

# (1) Create the dataset from the raw file
dataset_name="molecules-3.5M"
if [ ! -d "datasets/$dataset_name" ]; then
    echo "[-] Generating the dataset..."
    python scripts/dataprocessing/create_sft_dataset.py \
        --json_dataset datasets/raw/$dataset_name.json \
        --output_dataset datasets/$dataset_name \
        --data_column_name smiles  $ablated_flag
fi
    
# (2) Train the base model with sft
if [ ! -d "models/mol-llama-1b" ]; then
    echo "[-] Training base model..."
    accelerate launch --config-file configs/accelerate.yaml scripts/sft/train.py --config configs/base_model.yaml
fi
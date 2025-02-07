#!/bin/bash

properties="JNK3 DRD2 GSK3B CYP2C19_Veith CYP2D6_Veith"

# Check ablated flag
ablated_flag=""
if [ "$1" == "--ablated" ]; then
    ablated_flag="--ablated"
fi

# (1) Create the dataset from the raw set
offline_dataset_name="ric-offline-molecules-3.5M"
offline_scores_name="ric-offline-scores-molecules-3.5M"
if [ ! -d "datasets/$offline_dataset_name" ]; then
    echo "[1] Generating the offline dataset..."
    python scripts/dataprocessing/create_ric_offline_dataset.py \
        --json_dataset datasets/raw/molecules-3.5M.json \
        --output_dataset datasets/$offline_dataset_name \
        --output_scores datasets/$offline_scores_name \
        --data_column_name prompt \
        --properties $properties \
        $ablated_flag
fi
    
# (2) Train the offline RiC model (entire dataset)
ric_offline_model=""models/ric-offline-mol-llama-1b""
if [ ! -d $ric_offline_model ]; then
    echo "[2] Training RiC offline model..."
    accelerate launch --config-file configs/accelerate.yaml scripts/sft/train.py --config configs/mol_llama_ric_offline.yaml
fi

# (1) Create the dataset from the raw set
online_dataset_name="ric-online-molecules-3.5M"
temp_dataset_name="datasets/temp-scores-molecules-3.5M"
if [ ! -d "datasets/$online_dataset_name" ]; then
    echo "[3] Generating the online dataset..."
    python scripts/dataprocessing/create_ric_online_dataset.py \
        --batch_size 16 --num_samples 100 \
        --seed 42 --data_column_name prompt \
        --properties $properties \
        --offline_dataset datasets/$offline_scores_name \
        --output_dataset datasets/$online_dataset_name \
        --ric_model $ric_offline_model \
        --temp_scores_dataset $temp_dataset_name
fi
rm -rf $temp_dataset_name

# (4) Train the online RiC model (pareto samples)
if [ ! -d "models/ric-online-mol-llama-1b" ]; then
    echo "[4] Training RiC online model..."
    accelerate launch --config-file configs/accelerate.yaml scripts/sft/train.py --config configs/mol_llama_ric_online.yaml
fi
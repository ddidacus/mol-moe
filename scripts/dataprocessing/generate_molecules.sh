#!/bin/bash

prompt="<JNK3=1.0><DRD2=1.0><GSK3B=1.0><CYP2D6_Veith=1.0><CYP2C19_Veith=1.0><s>"
model="models/mol-moe-6x1b"

# MoE
mkdir -p results
python scripts/dataprocessing/generate_molecules.py \
    --seed 42 --model_name $model \
    --n_samples 512 --batch_size 512 --tasks JNK3 DRD2 GSK3B CYP2D6_Veith CYP2C19_Veith \
    --device cuda --prompt $prompt --out_molecules results/molecules

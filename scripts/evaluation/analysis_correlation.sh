#!/bin/bash

seed=42
N=256
batch_size=256
n_points=10
temperature=1.0
top_p=0.9

# Rewarded soups
for task in JNK3 DRD2 GSK3B CYP2D6 CYP2C19; do
    python scripts/evaluation/score_correlation.py \
        --seed $seed --n_generations $N --batch_size $batch_size --n_points $n_points \
        --method "rs" --output results/rs \
        --model_name models/mol-llama-1b --merging_method linear \
        --task $task
done
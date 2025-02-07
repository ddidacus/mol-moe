#!/bin/bash

# MoE
mkdir -p results/mol-moe
accelerate launch \
    --main_process_port 23457 --config_file configs/accelerate.yaml scripts/evaluation/score_steerability.py \
    --seed 42 --n_generations 2048 --batch_size 512 --n_steerings 10 \
    --method "mol-moe" --output results/mol-moe \
    --model models/mol-moe-6x1b --verbose 1

# RS
mkdir -p results/rs
accelerate launch \
    --main_process_port 23457 --config_file configs/accelerate.yaml scripts/evaluation/score_steerability.py \
    --seed 42 --n_generations 2048 --batch_size 512 --n_steerings 10 \
    --method "rs" --output results/rs \
    --merging_method linear \
    --model models/mol-llama-1b

# RiC
mkdir -p results/ric
accelerate launch \
    --main_process_port 23457 --config_file configs/accelerate.yaml scripts/evaluation/score_steerability.py \
    --seed 42 --n_generations 2048 --batch_size 512 --n_steerings 10 \
    --method "ric" --output results/ric \
    --model models/ric-online-mol-llama-1b



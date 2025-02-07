#!/bin/bash

prompt_scores_in_ctx="<JNK3=1.0><DRD2=1.0><GSK3B=1.0><CYP2D6_Veith=1.0><CYP2C19_Veith=1.0><s>"
prompt_standard="<s>"

mkdir -p results

for seed in {42..62}
do
    # MoE
    mkdir -p results/mol-moe
    python scripts/evaluation/score_sample_quality.py \
        --seed $seed --model_name models/mol-moe-6x1b \
        --n_samples 2048 --batch_size 512 --tasks JNK3 DRD2 GSK3B CYP2D6_Veith CYP2C19_Veith \
        --device cuda --out_name results/mol-moe/seed=$seed-samples.json \
        --prompt $prompt_scores_in_ctx

    # RLHF
    for task in JNK3 DRD2 GSK3B CYP2C19_Veith CYP2D6_Veith; do
        task="CYP2D6_Veith"
        mkdir -p results/expert_$task
        python scripts/evaluation/score_sample_quality.py \
            --seed $seed \
            --model_name models/mol-llama-1b-$task/consolidated \
            --n_samples 512 \
            --batch_size 512 \
            --tasks JNK3 DRD2 GSK3B CYP2C19_Veith CYP2D6_Veith \
            --device cuda \
            --out_name results/expert_$task/seed=$seed-samples.json \
            --prompt $prompt_standard
    done

    # MORLHF
    mkdir -p results/morlhf
    python scripts/evaluation/score_sample_quality.py \
        --seed $seed --model_name models/morlhf-mol-llama-1b \
        --n_samples 2048 --batch_size 512 --tasks CYP2D6_Veith CYP2C19_Veith DRD2 JNK3 \
        --device cuda --out_name results/morlhf/seed=$seed-samples.json \
        --prompt $prompt_standard
    
    # RS
    mkdir -p results/rs
    python scripts/evaluation/score_sample_quality.py \
        --seed $seed \
        --model_name models/rs-mol-llama-1b \
        --n_samples 2048 \
        --batch_size 512 \
        --tasks JNK3 DRD2 GSK3B CYP2C19_Veith CYP2D6_Veith \
        --device cuda \
        --out_name results/rs/seed=$seed-samples.json \
        --prompt $prompt_standard
    
    # RiC
    mkdir -p results/ric
    python scripts/evaluation/score_sample_quality.py \
        --seed $seed \
        --model_name models/ric-online-mol-llama-1b \
        --n_samples 2048 --batch_size 512 --tasks JNK3 DRD2 GSK3B CYP2C19_Veith CYP2D6_Veith \
        --device cuda --out_name results/ric/seed=$seed-samples.json \
        --prompt $prompt_scores_in_ctx

done

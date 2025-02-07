#!/bin/bash

# (1)-(2)-(3)
bash scripts/train_experts.sh

# (4) Merge the expert models
python scripts/merging/create_rewarded_soups.py \
    --models \
        models/mol-llama-1b-JNK3/consolidated \
        models/mol-llama-1b-DRD2/consolidated \
        models/mol-llama-1b-GSK3B/consolidated \
        models/mol-llama-1b-CYP2C19_Veith/consolidated \
        models/mol-llama-1b-CYP2D6_Veith/consolidated \
    --coefficients 0.2 0.2 0.2 0.2 0.2 \
    --output_path models/rs-mol-llama-1b
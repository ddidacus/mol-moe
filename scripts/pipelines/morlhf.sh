#!/bin/bash

# (1)-(2)
bash scripts/train_base_model.sh

# (3) Train the expert model with MORL
accelerate launch --config-file configs/accelerate.yaml scripts/rl/rloo.py --config configs/mol_llama_morlhf.yaml


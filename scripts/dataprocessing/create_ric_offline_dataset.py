import os
import json
import argparse
from tqdm import tqdm
from tdc import Oracle
from multiprocessing import Pool
import multiprocessing
import pandas as pd
from datasets import Dataset
from molgen.rewards.ml_reward import ClassificationReward
from molgen.utils.constants import SPECIAL_CHARS
from typing import List, Tuple
import numpy as np

MIN_VALID_MOL_LEN = 8
N_CPUS = multiprocessing.cpu_count()

def score_molecules(smiles_list:List[str]) -> List[dict]:
    """Process a single molecule and return its scores."""
    mols = []
    for m in smiles_list:
        _s = {}
        for name, oracle in oracles.items():
            _s[name] = oracle(m)
        mols.append(_s)
    return mols

def format_smiles(mol:str, score:dict) -> str:
    """Format a SMILES string."""
    task_order = np.random.permutation(list(quantiles.keys()))
    prompt = ""
    for task in task_order:
        prompt += f"<{task}={score[task]:.2f}>"
    prompt += f"{SPECIAL_CHARS['bos']}{mol}{SPECIAL_CHARS['eos']}"
    return prompt

def process_smiles(smiles_list:List[str]) -> Tuple[List[str], List[dict]]:
    """Process a raw list of smiles"""
    # Parallel scoring of the molecules
    with Pool(processes=N_CPUS) as pool:
        scored_smiles = pool.map(score_molecules, [smiles_list[i:i+1000] for i in tqdm(range(0, len(smiles_list), 1000))])
    scored_smiles = [s for b in scored_smiles for s in b]
    for score_x, smiles_x in zip(scored_smiles, smiles_list): score_x["smiles"] = smiles_x

    # Filter molecules and create prompts simultaneously
    if ablated:
        filtered_prompts = []
        filtered_scores = []
        for mol, score in zip(smiles_list, scored_smiles):
            passes_thresholds = True
            if not ablated:
                for task in quantiles:
                    if score[task] >= quantiles[task]["threshold"]:
                        passes_thresholds = False
                        break
            if not ablated or passes_thresholds:
                prompt = format_smiles(mol, score)
                filtered_prompts.append(prompt)
                filtered_scores.append(score)

        smiles_scores = filtered_scores
        smiles_molecules = filtered_prompts
    else:
        smiles_scores = scored_smiles
        smiles_molecules = [format_smiles(x, score) for x, score in zip(smiles_list, scored_smiles)]

        return smiles_molecules, smiles_scores

def main(args:argparse.Namespace):
    
    global quantiles, oracles, ablated
    ablated = args.ablated
    quantiles = {
        "JNK3": {"threshold": 0.6},
        "DRD2": {"threshold": 0.6},
        "GSK3B": {"threshold": 0.6},
        "CYP2D6_Veith": {"threshold": 0.6},
        "CYP2C19_Veith": {"threshold": 0.6},
    }

    # Load scoring functions
    oracle_names = args.properties
    oracles = {}
    for t in oracle_names:
        if t in ["JNK3", "DRD2", "GSK3B", "LogP", "SA"]:
            oracles[t] = Oracle(t)
        elif t in ["CYP2D6_Veith", "CYP2C19_Veith"]:
            oracles[t] = ClassificationReward(os.path.join("support", f"{t}.pkl"))
        else:
            raise Exception(f"Unsupported task: {t}")

    # Load and process molecules
    print("[-] Loading molecules")
    with open(args.json_dataset) as f:
        molecules = json.load(f)["samples"]

    print("[-] Scoring and filtering molecules...")
    prompts, scores = process_smiles(molecules)

    print(f"[-] Found {len(prompts)} molecules")
    
    # Save datasets
    dataset = Dataset.from_dict({args.data_column_name: prompts})
    dataset.save_to_disk(args.output_dataset)

    dataset = Dataset.from_dict({args.data_column_name: scores})
    dataset.save_to_disk(args.output_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='python create_ric_offline_dataset.py')
    parser.add_argument('--json_dataset', help="path to a json list of raw samples", type=str, required=True)
    parser.add_argument('--data_column_name', help="name of the column containing the sample content, e.g. text", type=str, default="text")
    parser.add_argument('--output_dataset', help="path to store the processed textual HF dataset", type=str, required=True)
    parser.add_argument('--output_scores', help="path to store the processed scored textual HF dataset", type=str, required=True)
    parser.add_argument('--ablated', type=bool, help="hold-out high quality samples", default=False)
    parser.add_argument('--properties', type=str, nargs="+", help="list of molecule properties to consider for scoring", required=True)
    args = parser.parse_args()
    main(args=args)
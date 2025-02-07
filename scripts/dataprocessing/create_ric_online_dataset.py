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
from molgen.utils.log import mprint
from molgen.utils.compute import is_master_rank
from typing import List, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_from_disk 
quantiles = {}
oracles = {}
tasks = None

def format_prompt(smiles:str, scores:dict) -> str:
    task_order = np.random.permutation(tasks)
    prompt = ""
    for task in task_order:
        prompt += f"<{task}={scores[task]:.2f}>"
    return prompt + SPECIAL_CHARS["bos"] + smiles + SPECIAL_CHARS["eos"]

def create_prompt(example:dict) -> dict:
    global quantiles
    scores = {task: example[args.data_column_name][task] for task in tasks}
    prompt = format_prompt(example[args.data_column_name]["smiles"], scores)
    example[args.data_column_name]["prompt"] = prompt
    return example

def extract_smiles(text:str) -> str:
    try:
        text = text.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "")
        return text.split(SPECIAL_CHARS["bos"])[1].split(SPECIAL_CHARS["eos"])[0].replace(" ", "")
    except:
        return None
    
def score_molecule(mol:str, task_name:str) -> float:
    import sys, os
    try:
        # Temporarily redirect stderr to devnull
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        if task_name in ["JNK3", "DRD2", "GSK3B", "LogP", "SA"]:
            oracle = Oracle(task_name)
        elif task_name in ["CYP2D6_Veith", "CYP2C19_Veith"]:
            oracle = ClassificationReward(os.path.join("support", f"{task_name}.pkl"))
        else:
            raise Exception(f"Unsupported task: {task_name}")
        
        result = oracle(mol)
        
        # Restore stderr
        sys.stderr = stderr
        return result
    except:
        # Restore stderr in case of exception
        if 'stderr' in locals():
            sys.stderr = stderr
        return None

def generate_pareto_prompts(args: argparse.Namespace) -> Dataset:
    mprint(f"[-] Creating {args.num_samples} prompts...")
    max_values = {t: float(m) for t, m in zip(args.properties, [1.0]*len(args.properties))}
    tasks = list(max_values.keys())
    prompts = []
    for _ in range(args.num_samples):
        # Choose random task to have non-max value
        random_task = np.random.choice(tasks)
        values = {}
        for task in tasks:
            if task == random_task:
                values[task] = np.random.uniform(0, max_values[task])
            else:
                values[task] = max_values[task]
        # Create prompt with random order of tasks
        task_order = np.random.permutation(tasks)
        prompt = ""
        for task in task_order:
            prompt += f"<{task}={values[task]:.2f}>"
        prompt += "<s>"
        prompts.append(prompt)
    return Dataset.from_dict({"prompt": prompts})

def generate_augmented_pareto(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, pareto_prompts: Dataset) -> List[str]:
    """ Generate samples from a dataset of pareto-conditioned prompts"""
    mprint("[-] Generating responses...")
    generation_kwargs = {
        "max_new_tokens": 128,
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": 1.0
    }
    responses = []
    for i in tqdm(range(0, len(pareto_prompts), args.batch_size)):
        batch = pareto_prompts[i:i+args.batch_size]["prompt"]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        outputs = outputs.cpu()
        responses += tokenizer.batch_decode(outputs)
    smiles_list = [extract_smiles(r) for r in responses]
    return smiles_list

def score_molecules(smiles_list:List[str]) -> List[dict]:
    """Process a single molecule and return its scores."""
    mols = []
    for m in smiles_list:
        _s = {}
        for name, oracle in oracles.items():
            _s[name] = oracle(m)
        _s["smiles"] = m
        mols.append(_s)
    return mols
    
def filter_scored_molecules(scored_molecules: Dataset, tasks: List[str], threshold: float) -> Dataset:
    global quantiles
    scores_by_task = {task: [x[args.data_column_name][task] for x in scored_molecules if all([v is not None for v in x[args.data_column_name].values()])] for task in tasks}
    quantiles = {task: np.quantile(scores_by_task[task], threshold) for task in tasks}
    filtered_dataset = scored_molecules.filter(lambda x: all([v is not None for v in x[args.data_column_name].values()]) and all(x[args.data_column_name][task] >= quantiles[task] for task in tasks))
    return filtered_dataset

def main(args:argparse.Namespace):
    global tasks, oracles
    tasks = args.properties
    pareto_prompts = generate_pareto_prompts(args)

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

    # ========== Pareto from offline dataset
    offline_dataset = load_from_disk(args.offline_dataset)
    mprint("[-] Filtering dataset molecules by scores...")
    filtered_offline_dataset = filter_scored_molecules(scored_molecules=offline_dataset, tasks=tasks, threshold=0.7) # Dataset
    offline_prompted_dataset = filtered_offline_dataset.map(
        create_prompt,
        num_proc=os.cpu_count()
    )
    # remove the scores to keep only the prompt with conditioning
    offline_prompted_dataset = [s[args.data_column_name]["prompt"] for s in offline_prompted_dataset]
    mprint(f"[-] # Pareto samples from data: {len(offline_prompted_dataset)}")

    # ========== Pareto from model augmentation
    mprint("[-] Setting up model for inference...")
    tokenizer = AutoTokenizer.from_pretrained(args.ric_model)
    model = AutoModelForCausalLM.from_pretrained(args.ric_model, trust_remote_code=True).to("cuda")

    augmented_smiles_list = generate_augmented_pareto(model=model, tokenizer=tokenizer, pareto_prompts=pareto_prompts)
    with Pool(processes=os.cpu_count()) as pool:
        scored_smiles = pool.map(score_molecules, [augmented_smiles_list[i:i+1000] for i in tqdm(range(0, len(augmented_smiles_list), 1000))])
    scored_augmented_molecules = [s for b in scored_smiles for s in b]
    scored_augmented_molecules = Dataset.from_dict({args.data_column_name: scored_augmented_molecules})

    mprint("[-] Filtering generated molecules by scores...")
    filtered_augmented_dataset = filter_scored_molecules(scored_molecules=scored_augmented_molecules, tasks=tasks, threshold=0.7) # Dataset
    augmented_prompted_dataset = filtered_augmented_dataset.map(
        create_prompt,
        num_proc=os.cpu_count()
    )
    # remove the scores to keep only the prompt with conditioning
    filtered_augmented_dataset = [s[args.data_column_name]["prompt"] for s in augmented_prompted_dataset]
    mprint(f"[-] # Pareto samples from model: {len(filtered_augmented_dataset)}")

    # ======= Combine and save
    online_dataset = offline_prompted_dataset + filtered_augmented_dataset
    online_dataset = Dataset.from_dict({args.data_column_name: online_dataset})
    mprint(f"[-] Total # pareto samples: {len(filtered_offline_dataset)}")

    # Saving datasets to disk
    if is_master_rank: online_dataset.save_to_disk(args.output_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='python create_ric_online_dataset.py')
    parser.add_argument('--batch_size', help="model inference batch size", type=int, required=True)
    parser.add_argument('--num_samples', help="generation seed, default: 100,000", type=int, default=42)
    parser.add_argument('--seed', help="size of the pareto front dataset", type=int, default=100000)
    parser.add_argument('--data_column_name', help="name of the column containing the sample content, e.g. text", type=str, default="text")
    parser.add_argument('--offline_dataset', help="path to the full offline dataset", type=str, required=True)
    parser.add_argument('--output_dataset', help="path to store the processed textual HF dataset", type=str, required=True)
    parser.add_argument('--properties', type=str, nargs="+", help="list of molecule properties to consider for scoring", required=True)
    parser.add_argument('--ric_model', type=str, help="path to the first stage RiC model", required=True)
    parser.add_argument('--temp_scores_dataset', help="path to store the temporary set of scored molecules", type=str, required=True)
    args = parser.parse_args()
    main(args=args)
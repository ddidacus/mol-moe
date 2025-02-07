from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from molgen.utils.constants import SPECIAL_CHARS
from tdc import Oracle, Evaluator
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import json
import re
import os
from molgen.rewards.ml_reward import ClassificationReward


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='python dump_molecules.py')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=1.0) 
    parser.add_argument('--n_samples', type=int, required=True)  
    parser.add_argument('--n_best', type=int, default=1)  
    parser.add_argument('--batch_size', type=int, required=True)  
    parser.add_argument('--out_molecules', type=str, default=None) 
    parser.add_argument('--device', type=str, required=True) 
    parser.add_argument('--train_set', type=str, default=None) 
    parser.add_argument('--tasks', type=str, nargs="+", default=["JNK3", "DRD2", "GSK3B", "CYP2C19", "CYP2D6"]) 
    parser.add_argument('--top_k', type=int, default=50) 
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--top_p', type=float, default=1.0) 
    parser.add_argument('--prompt', type=str, default=SPECIAL_CHARS["bos"])
    parser.add_argument('--parsing', type=str, choices=["mol-moe", "instructed"], default="mol-moe")
    args = parser.parse_args()

    # Loading training distribution
    if args.train_set:
        with open(args.train_set) as f:
            train_set = json.load(f)
            f.close()

    # Loading the model
    print(f"[-] Run seed: {args.seed}")
    set_seed(args.seed)
    device = args.device

    # Preparing oracles
    oracles = {}
    for t in args.tasks:
        if t in ["JNK3", "DRD2", "GSK3B", "LogP", "SA"]:
            oracles[t] = Oracle(t)
        elif t in ["CYP2D6_Veith", "CYP2C19_Veith"]:
            oracles[t] = ClassificationReward(os.path.join("support", f"{t}.pkl"))
        else:
            raise Exception(f"Unsupported task: {t}")
    eval_uniqueness = Evaluator("uniqueness")
    eval_diversity = Evaluator("diversity")

    # Prepare model soup
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    generation_kwargs = {
        "max_new_tokens": 128,
        "min_length": -1,
        "top_k": 0.0,
        "top_p": args.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": args.temperature
    }

    # Inference
    scores = {}
    samples = []
    tot_gen_mols = []
    scores["avg"] = []
    molecules_data = []  # List to store molecule data for CSV
    iterations = args.n_samples // args.batch_size
    prompt = args.prompt
    print(f"[-] Prompt: {prompt}")
    unparsed = 0

    for itx in tqdm(range(iterations)):
        toks = tokenizer([prompt]*args.batch_size, return_tensors="pt")["input_ids"].to(device)
        prompt_responses = model.generate(toks, **generation_kwargs)
        prompt_responses = tokenizer.batch_decode(prompt_responses)
        # Parsing
        gen_mols = []
        for pr in prompt_responses:
            try:
                if args.parsing == "instructed":
                    parsed = pr.split("Answer: ")[1].split("[END]")[0].replace(" ", "")
                    gen_mols.append(parsed)
                else:
                    parsed = pr.split(SPECIAL_CHARS["bos"])[1].split(SPECIAL_CHARS["eos"])[0].replace(" ", "")
                    gen_mols.append(parsed)
            except: 
                unparsed += 1
        tot_gen_mols += gen_mols

        # Scoring
        for mol in gen_mols:
            mol_scores = {}  # Dictionary to store scores for this molecule
            mol_avg = []
            for key in oracles:
                if key not in scores: scores[key] = []
                score = oracles[key](mol)
                scores[key].append(score)
                mol_avg.append(score)
                mol_scores[key] = score
            
            samples.append(mol_avg)
            avg_score = np.mean(mol_avg)
            scores["avg"].append(avg_score)
            
            # Store molecule data for CSV
            mol_data = {
                'molecule': mol,
                'average_score': avg_score
            }
            # Add individual task scores
            mol_data.update({f'score_{key}': value for key, value in mol_scores.items()})
            molecules_data.append(mol_data)

    # Create and save CSV
    df = pd.DataFrame(molecules_data)
    csv_filename = os.path.splitext(args.out_molecules)[0] + '.csv' if args.out_molecules else 'molecules_with_scores.csv'
    df.to_csv(csv_filename, index=False)
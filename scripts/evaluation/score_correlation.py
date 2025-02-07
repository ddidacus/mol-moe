from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from molgen.rewards.ml_reward import ClassificationReward
import multiprocessing as mp
from datasets import Dataset
from tdc import Oracle
from tqdm import tqdm
import numpy as np
import argparse
import torch
import json
import sys
import os

def init_LM(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer

def load_and_merge_models(model_name, task, weights, method="linear"):
    
    # ============= Merge linearly
    if method == "linear":
        # Load all models
        print("[-] Merging linearly")
        model_paths = [
            os.path.join(f"{model_name}-{task}", "consolidated"),
            os.path.join(f"{model_name}", "consolidated")
        ]
        models_weights = []
        for m in model_paths:
            model = AutoModelForCausalLM.from_pretrained(m, device_map='cpu')
            models_weights.append(model.state_dict())
            del model
            torch.cuda.empty_cache()
        
        # Get tokenizer from first model
        tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
        
        # Average weights
        for key in models_weights[0]:
            models_layers = [m[key] for m in models_weights]
            weights_sum = None
            for i, k in enumerate(weights):
                curr_w = k * models_layers[i]
                if weights_sum is None:
                    weights_sum = curr_w
                else:
                    weights_sum += curr_w
            # Use first model for merge by default
            models_weights[0][key] = weights_sum
        
        # Load merged weights into model
        model = AutoModelForCausalLM.from_pretrained(model_paths[0], device_map='cpu')
        model.load_state_dict(models_weights[0])
    
    # ============= Merge with mergekit
    else:
        # Define paths and tasks
        print(f"[-] Merging with {method}")
        merged_path = model_name

        # Build mergekit YAML config
        yaml_config = "models:\n"
        path = os.path.join(f"{model_name}-{task}", "consolidated")
        yaml_config += f"  - model: {path}\n"
        yaml_config += "    parameters:\n"
        yaml_config += f"      weight: {weights[i]:.6f}\n"

        # Add base model and merge settings
        base_path = model_name
        yaml_config += f"base_model: {base_path}\n"
        yaml_config += f"merge_method: {method}\n"
        yaml_config += "dtype: float16"

        # Write config and run mergekit
        config_path = os.path.join("configs", "merging", "eval_config.yml")
        with open(config_path, "w") as f:
            f.write(yaml_config)

        print(f"[+] Weights: {weights}")
        merging_process = os.system(f"mergekit-yaml --verbose --allow-crimes {config_path} {merged_path} >/dev/null 2>&1")
        if merging_process != 0: raise Exception("Error in merging")
        
        # Load merged model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(merged_path)
        tokenizer = AutoTokenizer.from_pretrained(merged_path)
    
    return model, tokenizer

def process_single_sample(sample):
    sample = sample["text"]
    oracles = {}
    t = args.task
    if t in ["JNK3", "DRD2", "GSK3B", "LogP", "SA"]:
        oracles[t] = Oracle(t)
    elif t in ["CYP2D6_Veith", "CYP2C19_Veith"]:
        oracles[t] = ClassificationReward(os.path.join("models", f"{t}.pkl"))
    else:
        raise Exception(f"Unsupported task: {t}")
    try:
        molecule = sample.split("<s>")[1].split("</s>")[0]
        scores = {task: oracle(molecule) for task, oracle in oracles.items()}
        return {"molecule": molecule, "scores": scores}
    except:
        return {"molecule": None, "scores": None}

def generate_samples(model, tokenizer, prompt, N, batch_size, generation_kwargs):
    model.eval()
    all_outputs = []
    
    num_batches = N // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), file=outstream):
            inputs = tokenizer([prompt] * batch_size, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model.generate(**inputs, **generation_kwargs)
            decoded = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            all_outputs.extend(decoded)
            
        remaining = N % batch_size
        if remaining > 0:
            inputs = tokenizer([prompt] * remaining, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model.generate(**inputs, **generation_kwargs)
            decoded = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            all_outputs.extend(decoded)
    
    return all_outputs

def correlations(args):

    # =========== Inference
    print("[-] Starting inference...")
    X_alphas = []
    Y_scores = []

    for alpha_1 in np.linspace(0, 1, args.n_points):

        alpha_2 = 1-alpha_1
        weights = np.ones(2)*alpha_2
        weights[0] = alpha_1 # first is always the expert model
        print(f"[-] Weights: {[round(x, 2) for x in list(weights)]}")

        model, tokenizer = load_and_merge_models(task=args.task,  model_name=args.model_name, weights=weights, method=args.merging_method)
        model.to(args.device)
        generation_kwargs = {
            "max_new_tokens": 128,
            "min_length": -1,
            "top_k": 0.0,   
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "temperature": args.temperature
        }
        prompt = "<s>"
        samples = generate_samples(model, tokenizer, prompt, args.n_generations, args.batch_size, generation_kwargs)
        del model, tokenizer
        torch.cuda.empty_cache()
            
        dataset = Dataset.from_dict({"text": samples})
        processed_dataset = dataset.map(
            process_single_sample,
            num_proc=mp.cpu_count(),
            remove_columns=["text"]
        )
        molecule_scores = [x["scores"][args.task] for x in processed_dataset if x["molecule"] is not None]
        molecule_scores = np.mean(molecule_scores)
        X_alphas.append(weights[0])
        Y_scores.append(molecule_scores)
        print(X_alphas)
        print(Y_scores)
        print("=========")

    # =========== Results
    correlation_matrix = np.corrcoef(X_alphas, Y_scores)
    print(correlation_matrix)

    with open(os.path.join(args.output, f"correlation_{args.task}.json"), "w") as f:
        json.dump({
            "correlation": correlation_matrix[0][1],
            "X": X_alphas,
            "Y": Y_scores
        }, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--method', type=str, choices=["ric", "rs", "rs_balanced", "moe"], required=True)
    parser.add_argument('--merging_method', type=str, choices=["linear", "breadcrumbs", "dare", "ties"])
    parser.add_argument('--n_generations', type=int, required=True)
    parser.add_argument('--n_test_samples', type=int, default=256)
    parser.add_argument('--n_points', type=int, required=True)
    parser.add_argument('--task', type=str, required=True)

    args = parser.parse_args()

    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.verbose:
        outstream = sys.stderr
    else:
        sys.stderr = open(os.devnull, 'w')
        outstream = sys.stdout

    correlations(args)
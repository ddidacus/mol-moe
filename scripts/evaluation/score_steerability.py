import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs
import multiprocessing as mp
from tdc import Oracle
from tqdm import tqdm
import numpy as np
import argparse
import torch
import torch.distributed as dist
import json
import os
import sys
from datasets import Dataset
from molgen.rewards.ml_reward import ClassificationReward

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

def init_LM(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = accelerator.prepare(model)
    return model, tokenizer

def load_and_merge_models(model_name, tasks, weights, method="linear"):
    
    # ============= Merge linearly
    if method == "linear":
        # Load all models
        if accelerator.is_main_process: print("[-] Merging linearly")
        model_paths = [os.path.join(f"{model_name}-{t}", "consolidated") for t in tasks]
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
        if accelerator.is_main_process: print(f"[-] Merging with {method}")
        merged_path = model_name

        # Build mergekit YAML config
        yaml_config = "models:\n"
        for i, task in enumerate(tasks):
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

        if accelerator.is_main_process: print(f"[+] Weights: {weights}")
        merging_process = os.system(f"mergekit-yaml --verbose --allow-crimes {config_path} {merged_path} >/dev/null 2>&1")
        if merging_process != 0: raise Exception("Error in merging")
        
        # Load merged model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(merged_path)
        tokenizer = AutoTokenizer.from_pretrained(merged_path)
    
    return model, tokenizer

def generate_control_points(N, norm_ranges, normalize=False):
    # Not used here, just for reference on how to generate the JSON files
    # Extract ranges for each dimension
    dimensions = list(norm_ranges.keys())
    ranges = np.array([(norm_ranges[d][1] - norm_ranges[d][0]) for d in dimensions])
    min_vals = np.array([norm_ranges[d][0] for d in dimensions])
    max_vals = np.array([norm_ranges[d][1] for d in dimensions])
    
    # Initialize array to store all points
    points = np.zeros((N, len(dimensions)))
    
    # Generate N points from Dirichlet distribution
    alpha = np.ones(len(dimensions))  # Uniform Dirichlet with alpha=1
    points = np.random.dirichlet(alpha, size=N)
    
    # Scale points to match ranges
    points = points * ranges + min_vals
    
    # Convert to dictionary format matching the dimensions
    result = []
    for point in points:
        if normalize: point = point / np.sum(point)
        result.append({dim: val for dim, val in zip(dimensions, point)})
        
    return result

def process_single_sample(sample):
    sample = sample["text"]
    tasks = ["JNK3", "DRD2", "GSK3B", "CYP2D6_Veith", "CYP2C19_Veith"]
    oracles = {}
    for t in tasks:
        if t in ["JNK3", "DRD2", "GSK3B"]:
            oracles[t] = Oracle(t)
        elif t in ["CYP2D6_Veith", "CYP2C19_Veith"]:
            oracles[t] = ClassificationReward(os.path.join("support", f"{t}.pkl"))
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
    
    # Calculate samples per GPU
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    samples_per_gpu = N // world_size
    start_idx = rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if rank != world_size-1 else N
    
    local_N = end_idx - start_idx
    num_batches = local_N // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), disable=not accelerator.is_local_main_process, file=outstream):
            inputs = tokenizer([prompt] * batch_size, return_tensors="pt")
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
            outputs = accelerator.unwrap_model(model).generate(**inputs, **generation_kwargs)
            decoded = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            all_outputs.extend(decoded)
            
        # Handle remaining samples
        remaining = local_N % batch_size
        if remaining > 0:
            inputs = tokenizer([prompt] * remaining, return_tensors="pt")
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
            outputs = accelerator.unwrap_model(model).generate(**inputs, **generation_kwargs)
            decoded = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            all_outputs.extend(decoded)
    
    # Gather results from all GPUs
    gathered_outputs = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_outputs, all_outputs)
    
    if accelerator.is_main_process:
        # Flatten gathered outputs
        final_outputs = []
        for outputs in gathered_outputs:
            final_outputs.extend(outputs)
        return final_outputs
    return None

def evaluate_steerability(args):
    
    # ============= Test steerings
    scaled_test_samples = json.load(open("support/scaled_test_samples.json"))
    unscaled_test_samples = json.load(open("support/unscaled_test_samples.json"))

    if args.method in ["rs", "mol-moe"]: test_samples = scaled_test_samples
    elif args.method == "ric": test_samples = unscaled_test_samples

    # ============= Models
    print("[-] Initializing models...")
    
    if args.method in ["morlhf", "mol-moe", "ric"]:
        model, tokenizer = init_LM(args.model)

    elif args.method == "rs":
        pass
    else:
        raise NotImplementedError("Unsupported method to evaluate")
    
    generation_kwargs = {
        "max_new_tokens": 128,
        "min_length": -1,
        "top_k": 0.0,   
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": 1.0
    }

    # ============= Prompts
    if args.method in ["ric", "mol-moe"]:
        prompts = []
        for sample in test_samples:
            prompt = "".join([f"<{task}={score:.2f}>" for task, score in zip(sample.keys(), sample.values())]) + "<s>"
            prompts.append(prompt)

    elif args.method in ["rs", "morlhf"]:
        prompts = ["<s>"] * args.n_steerings

    else:
        raise NotImplementedError("Unsupported method to evaluate")

    # ========= Inference 
    errors = []
    pareto = []
    expected = []

    # For each test sample
    print("[-] Starting inference...")
    for i in range(args.n_steerings):
        
        # Rewarded soups
        if args.method == "rs":
            # Create model for this test sample
            weights = np.array(list(test_samples[i].values()))
            model, tokenizer = load_and_merge_models(model_name=args.model, tasks=args.tasks, weights=weights, method=args.merging_method)
            # Move model to GPU only when needed
            model = accelerator.prepare(model)
            samples = generate_samples(model, tokenizer, prompts[i], args.n_generations, args.batch_size, generation_kwargs)
            model = accelerator.free_memory(model)
            del model, tokenizer
            torch.cuda.empty_cache()


        # Rewards in Context
        elif args.method in ["ric", "mol-moe"]:
            weights = np.array(list(test_samples[i].values()))
            samples = generate_samples(model, tokenizer, prompts[i], args.n_generations, args.batch_size, generation_kwargs)
            
        # Parse & score outputs batch
        if accelerator.is_main_process:
            dataset = Dataset.from_dict({"text": samples})
            processed_dataset = dataset.map(
                process_single_sample,
                num_proc=mp.cpu_count(),
                remove_columns=["text"]
            )
            outputs = [x for x in processed_dataset if x["molecule"] is not None]

            # Compute the error
            outputs_samples = np.array([list(x["scores"].values()) for x in outputs])
            task_stats = [[outputs_samples[:, dim].min(), outputs_samples[:, dim].max()] for dim in range(outputs_samples.shape[1])]
            y_hat = outputs_samples.mean(axis=0) # score
            y = np.array(list(scaled_test_samples[i].values()))

            # (1) Store pareto front
            pareto.append(y_hat)

            # (2) Compute error
            task_errors = np.array([np.absolute(y[t]-y_hat[t]) for t in range(y_hat.shape[0])])
            errors.append(task_errors)
            
            # Store expected points
            expected.append(y) 
        
        # Save to cache
        if accelerator.is_main_process:
            print(f"[-] MAE: {np.array(errors).mean(0)}")
            if not os.path.exists(args.output_path): os.mkdir(args.output_path)
            with open(os.path.join(args.output_path, "pareto.json"), 'w') as f:
                json.dump({
                    "pareto_y_hat": [p.tolist() for p in pareto],
                    "pareto": [p.tolist() for p in expected]
                }, f)
            with open(os.path.join(args.output_path, "steerability.json"), 'w') as f:
                json.dump({"mae": list(np.array(errors).mean(0)), "errors": [list(e) for e in errors]}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--method', type=str, choices=["ric", "rs", "morlhf", "mol-moe"], required=True)
    parser.add_argument('--merging_method', type=str, choices=["linear", "breadcrumbs", "dare", "ties"], default="linear")
    parser.add_argument('--tasks', help="merging coefficients", type=str, nargs="+", default=["JNK3", "DRD2", "GSK3B", "CYP2D6_Veith", "CYP2C19_Veith"])
    parser.add_argument('--n_generations', type=int, required=True)  # Changed to int type
    parser.add_argument('--n_steerings', type=int, required=True)  # Added argument
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize accelerator
    if accelerator.is_main_process: print(f"[-] Run seed: {args.seed}")

    # Suppress stderr
    if args.verbose:
        outstream = sys.stderr
    else:
        sys.stderr = open(os.devnull, 'w') # silence logs
        outstream = sys.stdout # tqdm stdout

    # Run evaluation
    evaluate_steerability(args)
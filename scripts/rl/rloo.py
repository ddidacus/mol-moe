from trl.trainer.rloo_trainer import RLOOConfig, RLOOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed, AdamW, PreTrainedTokenizer, TrainerCallback, GenerationConfig   
from accelerate import Accelerator
from datasets import Dataset
from tdc import Oracle
import argparse
import wandb
import json
import time
import os
import re
from molgen.utils.rlhf_toolkit import *
from molgen.rewards.singletask_reward import *
from molgen.config.rlhf_config import RLOOTrainingConfig
from molgen.rewards.morlhf_reward import MORLHFRewardModel
from molgen.rewards.moe_reward import MoEMaximizationRM
from molgen.utils.training import set_trainable_routers_only

def prepare_rlhf_inputs(config, tokenizer):
        def tokenize(element):
            outputs = tokenizer(
                element["query"],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}
        generation_inputs = [SPECIAL_CHARS["bos"]] * config.num_samples
        data_dict = {"query": generation_inputs}
        dataset = Dataset.from_dict(data_dict)
        train_dataset = dataset.map(tokenize, batched=True, remove_columns="query")
        generation_inputs = [SPECIAL_CHARS["bos"]] * int(config.num_samples * 0.001)
        data_dict = {"query": generation_inputs}
        dataset = Dataset.from_dict(data_dict)
        eval_dataset = dataset.map(tokenize, batched=True, remove_columns="query")
        return train_dataset, eval_dataset


def main(config: RLOOTrainingConfig):
    accelerator = Accelerator()

    # ======== Model
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(config.base_model)
    ref_policy = AutoModelForCausalLM.from_pretrained(config.base_model)

    generation_kwargs = {
        "max_new_tokens": config.max_output_len,
        "min_length": -1,
        "top_k": 0.0,
        "top_p": config.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": config.temperature,
        "num_return_sequences": 1
    }
    policy.generation_config = GenerationConfig(**generation_kwargs)
    ref_policy.generation_config = GenerationConfig(**generation_kwargs)

    if config.is_moe:
        set_trainable_routers_only(policy)

    # Resume model if requested
    if config.enable_resume:
        run_path, do_resume_ckpt = load_run_folder(path=config.output_path)
        if do_resume_ckpt:
            config.model = run_path # load from checkpoint
            match = re.search(r'\b\d+\b', run_path)
            if match:
                starting_step = int(match.group())
                if accelerator.is_main_process:
                    print(f"[!] Resuming from step {starting_step}")
            else: starting_step = 0
        else:
            starting_step = 0
    else:
        do_resume_ckpt = False
        starting_step = 0
    
    # ======== Data loading
    train_dataset, eval_dataset = prepare_rlhf_inputs(config, tokenizer)

    # ======== Training
    early_stopping_callback = EarlyStoppingCallback(
        metric="objective/scores",
        patience=config.patience,
        min_delta=config.min_delta
    )
    accumulation_steps = config.batch_size // config.max_gpu_batch_size

    if len(config.tasks) == 1:
        reward_model = SingleTaskRewardModel(config.tasks[0], tokenizer)
    elif config.is_moe:
        reward_model = MoEMaximizationRM(tokenizer=tokenizer, batch_size=config.batch_size, tasks=config.tasks)
    else:
        reward_model = MORLHFRewardModel(config.tasks, tokenizer)
    
    trainer = BetaRLOOTrainer(
        max_checkpoints=config.max_checkpoints,
        starting_beta=config.beta,
        config=RLOOConfig(
            run_name=config.run_name,
            output_dir=config.output_path,
            seed=config.seed,
            learning_rate=config.lr,
            kl_coef=config.beta,
            per_device_train_batch_size=config.max_gpu_batch_size,
            gradient_accumulation_steps=accumulation_steps,
            total_episodes=len(train_dataset),
            logging_strategy="steps",
            eval_strategy="steps",
            save_strategy="steps",
            logging_steps=config.logging_steps,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            load_best_model_at_end=True,
            greater_is_better=True,
            metric_for_best_model="objective/scores"
        ),
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.add_callback(early_stopping_callback)
    trainer.train()

    # Save final model
    policy.save_pretrained(os.path.join(config.output_path, "consolidated"))
    tokenizer.save_pretrained(os.path.join(config.output_path, "consolidated"))
    

if __name__ == "__main__":

    # Load training run
    parser = argparse.ArgumentParser(prog='python rloo.py')
    parser.add_argument('--config', help="path to yaml config file", type=str, required=True)
    args = parser.parse_args()
    config = RLOOTrainingConfig()
    config.load_config(path=args.config)
    set_seed(config.seed)

    # Prepare logging
    if is_master_rank() and config.wandb_project_name:
        wandb.init(
            project=config.wandb_project_name, 
            name=config.output_model_name,
            config=config.to_dict()
        )

    main(config)

    
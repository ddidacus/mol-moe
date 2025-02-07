from molgen.utils.compute import load_run_folder, print_trainable_parameters
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from molgen.utils.constants import SPECIAL_CHARS
from molgen.config.sft_config import SFTTrainingConfig
from molgen.utils.log import mprint
from molgen.utils.compute import is_master_rank
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
import argparse
import wandb
import os

def main(config: SFTTrainingConfig):
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
    # ======== Data loading
    train_dataset = load_from_disk(config.hf_dataset)
    mprint(f"[-] Training on {len(train_dataset)} molecules")

    # ======== Model
    model = AutoModelForCausalLM.from_pretrained(config.base_model, trust_remote_code=True)
    print_trainable_parameters(model)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.add_special_tokens({'pad_token': SPECIAL_CHARS["pad"]})

    # Resume model if requested
    if config.enable_resume:
        run_path, do_resume_ckpt = load_run_folder(path=config.output_path)
        if do_resume_ckpt:
            mprint(f"[+] Resuming from checkpoint {run_path}")
            resume_checkpoint = run_path
        else:
            resume_checkpoint = None
    else: resume_checkpoint = None

    # ======== Training
    n_cpus = os.cpu_count()
    accumulation_steps = config.batch_size // config.max_gpu_batch_size
    train_config = SFTConfig(
        packing=config.packing,
        run_name=config.run_name,
        per_device_train_batch_size=config.max_gpu_batch_size,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        dataloader_num_workers=n_cpus,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.lr,
        output_dir=config.output_path,
        dataset_text_field=config.hf_dataset_text_field,
        max_seq_length=config.max_seq_len,
        gradient_accumulation_steps=accumulation_steps
    )
    trainer = SFTTrainer(
        model=model,
        args=train_config,
        train_dataset=train_dataset
    )
    # Training
    mprint("[-] Training...")
    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train(resume_from_checkpoint=None)
    
    # Saving the final model
    final_path = os.path.join(config.output_path, "consolidated")
    trainer.save_model(final_path)

if __name__ == "__main__":

    # Load training run
    parser = argparse.ArgumentParser(prog='python train.py')
    parser.add_argument('--config', help="path to yaml config file", type=str, required=True)
    args = parser.parse_args()
    config = SFTTrainingConfig()
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

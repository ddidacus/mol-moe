from dataclasses import dataclass, field
from typing import Optional, List, get_type_hints
import dataclasses
import yaml

@dataclass
class TrainingConfig:
    wandb_project_name: str = None
    run_name: str = None
    output_model_name: str = None
    output_path: str = None
    lr: float = None
    base_model: str = None
    batch_size: int = None
    max_gpu_batch_size: int = None
    seed: int = 42
    lr_scheduler_type: str = "linear"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    epochs: int = 1
    enable_resume: bool = True
    max_seq_len: int = 256
    warmup_steps: int = 10
    packing: bool = True
    eval_steps: int = 10
    logging_steps: int = 10
    save_steps: int = 10

    def save_config(self, path):
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f)

    def load_config(self, path):
        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            type_hints = get_type_hints(self.__class__)
            
            for key, value in config.items():
                if key not in type_hints:
                    continue
                    
                expected_type = type_hints[key]
                
                # Handle None values
                if value is None:
                    setattr(self, key, None)
                    continue
                
                try:
                    # Convert value to expected type
                    if expected_type == bool:
                        # Special handling for boolean values
                        if isinstance(value, str):
                            value = value.lower() in ('true', '1', 'yes', 'on')
                        else:
                            value = bool(value)
                    else:
                        value = expected_type(value)
                    
                    setattr(self, key, value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Failed to convert '{key}' value '{value}' to type {expected_type}: {str(e)}")

    def to_dict(self):
        return dataclasses.asdict(self)
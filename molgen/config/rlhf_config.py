from dataclasses import dataclass
from .training_config import TrainingConfig
from typing import List

@dataclass
class RLOOTrainingConfig(TrainingConfig):
    patience: int = 50
    min_delta: float = 0.025
    beta: float = 0.05
    tasks: list = None
    max_output_len: int = 128
    top_p: float = 0.9
    temperature:float = 1.0
    max_checkpoints: int = 4
    num_samples: int = None
    is_moe: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
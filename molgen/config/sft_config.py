from dataclasses import dataclass
from .training_config import TrainingConfig

@dataclass
class SFTTrainingConfig(TrainingConfig):
    hf_dataset_text_field: str = "text"
    hf_dataset: str = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
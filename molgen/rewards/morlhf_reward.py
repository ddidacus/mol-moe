import numpy as np
import torch
import os
from collections import deque
from typing import List, Optional
from molgen.utils.constants import SPECIAL_CHARS, TASK_TOKENS
from molgen.rewards.ml_reward import ClassificationReward
from molgen.utils.rlhf_toolkit import *
from molgen.utils.compute import is_master_rank
from typing import Optional
import torch.nn as nn
from tdc import Oracle
    
class MORLHFRewardModel(nn.Module):
    def __init__(self, tasks, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.base_model_prefix = "prefix"
        self.prefix = self
        self.tasks = tasks
        self.oracles = {}
        for t in tasks:
            if t in ["JNK3", "DRD2", "GSK3B", "LogP", "SA"]:
                self.oracles[t] = Oracle(t)
            elif t in ["CYP2D6_Veith", "CYP2C19_Veith"]:
                self.oracles[t] = ClassificationReward(os.path.join("support", f"{t}.pkl"))
            else:
                raise Exception(f"Unsupported task: {t}")

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        # Mimic the behavior of a language model backbone
        hidden_states = self.process_input(input_ids, attention_mask)
        return AttrDict({"hidden_states": hidden_states})

    def process_input(self, input_ids, attention_mask):
        # Use a fixed hidden_dim (e.g., 768) to ensure consistent size
        hidden_dim = 768
        max_len = input_ids.shape[1]
        hidden_states = torch.zeros((input_ids.shape[0], max_len, hidden_dim), device=input_ids.device)
        for i, (text, mask) in enumerate(zip(input_ids.tolist(), attention_mask)):
            # Use attention_mask to determine the actual sequence length
            seq_len = mask.sum().item()
            # Encode in a fake embedding the tokens (for any seq len, populate the first logit, rest is zeros)
            encoded = text[:seq_len]
            encoded = encoded[:seq_len] + [0] * (seq_len - len(encoded))
            hidden_states[i, :seq_len, 0] = torch.tensor(encoded, device=input_ids.device)
        return [hidden_states]  # Return as a list to match the expected format

    def score(self, hidden_states):
        # Extracting & parsing output tokens
        batch_size, seq_len, _ = hidden_states.shape
        tokens = [[int(t) for t in seq] for seq in hidden_states[:, :, 0].tolist()]
        texts = self.tokenizer.batch_decode(tokens)
        parsed_texts = []
        for t in texts:
            try:
                parsed_texts.append(t.split(SPECIAL_CHARS["bos"])[1].split(SPECIAL_CHARS["eos"])[0])
            except: parsed_texts.append(t)
        # Scoring
        scores = []
        for text in parsed_texts:
            try:
                avg_score = np.mean([scoring_fn(text) for scoring_fn in self.oracles.values()])
                scores.append(avg_score)
            except: scores.append(0.0)
        scores = torch.tensor(scores, device=hidden_states.device)
        return scores.unsqueeze(1).expand(-1, seq_len)  # Shape: (batch_size, seq_len)

    def modules(self):
        return []

    def to(self, _):
        return self


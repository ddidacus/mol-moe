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

class MoEMaximizationRM(nn.Module):
    def __init__(self, tokenizer, tasks, batch_size):
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

        self.score_queues = {task: deque(maxlen=4*batch_size) for task in TASK_TOKENS.keys()} # Store 4 batches worth of scores

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
        scores = []
        for t in texts:
            try:
                # Parsing the prompted weights (w)
                # make sure the prompted tasks (from dataset) match with the ones in the set of scoring oracles
                w = t.split(SPECIAL_CHARS["bos"])[0].replace("<|begin_of_text|>", "")
                w = {x.split("=")[0]: float(x.split("=")[1].replace(">", "")) for x in w.split("<")[1:] if x.split("=")[0] in list(self.oracles.keys())}
                generated_molecule = t.split(SPECIAL_CHARS["bos"])[1].split(SPECIAL_CHARS["eos"])[0]
                # Computing the predicted weights (w_hat)
                maximization_reward = 0
                task_scores = []
                for task in w.keys():
                    score = self.oracles[task](generated_molecule)
                    # Normalizing the scores
                    task_scores.append(score)
                    maximization_reward += w[task] * score # prompted weight
                assert not np.isnan(maximization_reward)
                scores.append(maximization_reward)
            except: scores.append(float(0)) 
        scores = torch.tensor(scores, device=hidden_states.device)
        return scores.unsqueeze(1).expand(-1, seq_len)  # Shape: (batch_size, seq_len)

    def modules(self):
        return []

    def to(self, _):
        return self


class MoEPrecisionRM(nn.Module):
    def __init__(self, tokenizer, tasks, batch_size):
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
                self.oracles[t] = ClassificationReward(os.path.join("models", f"{t}.pkl"))
            else:
                raise Exception(f"Unsupported task: {t}")

        self.score_queues = {task: deque(maxlen=4*batch_size) for task in TASK_TOKENS.keys()} # Store 4 batches worth of scores

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
        scores = []
        for t in texts:
            try:
                # Extracting the molecule and the prompted PROPORTIONS
                w = t.split(SPECIAL_CHARS["bos"])[0].replace("<|begin_of_text|>", "")
                w = {x.split("=")[0]: float(x.split("=")[1].replace(">", "")) for x in w.split("<")[1:] if x.split("=")[0] in list(self.oracles.keys())}
                generated_molecule = t.split(SPECIAL_CHARS["bos"])[1].split(SPECIAL_CHARS["eos"])[0]
                # Computing the precision reward (MAE)
                w_true = np.array([w for w in w.values()])
                scores_hat = np.array([self.oracles[task](generated_molecule) for task in w.keys()])
                w_hat = scores_hat / scores_hat.sum()
                score = 1.0 - np.absolute(w_true - w_hat).mean()
                assert not np.isnan(score)
                scores.append(score)
            except: scores.append(float(0)) 
        scores = torch.tensor(scores, device=hidden_states.device)
        return scores.unsqueeze(1).expand(-1, seq_len)  # Shape: (batch_size, seq_len)

    def modules(self):
        return []

    def to(self, _):
        return self


def MoEPrecisionReward(completions, **kwargs):
    # oracles
    oracles = {}
    for t in ["JNK3", "DRD2", "GSK3B", "CYP2D6_Veith", "CYP2C19_Veith"]:
        if t in ["JNK3", "DRD2", "GSK3B", "LogP", "SA"]:
            oracles[t] = Oracle(t)
        elif t in ["CYP2D6_Veith", "CYP2C19_Veith"]:
            oracles[t] = ClassificationReward(os.path.join("models", f"{t}.pkl"))
        else:
            raise Exception(f"Unsupported task: {t}")
    scores = []
    for t in completions:
        try:
            # Extracting the molecule and the prompted PROPORTIONS
            w = t.split(SPECIAL_CHARS["bos"])[0].replace("<|begin_of_text|>", "")
            w = {x.split("=")[0]: float(x.split("=")[1].replace(">", "")) for x in w.split("<")[1:] if x.split("=")[0] in list(oracles.keys())}
            generated_molecule = t.split(SPECIAL_CHARS["bos"])[1].split(SPECIAL_CHARS["eos"])[0]
            # Computing the precision reward (MAE)
            w_true = np.array([w for w in w.values()])
            scores_hat = np.array([oracles[task](generated_molecule) for task in w.keys()])
            w_hat = scores_hat / scores_hat.sum()
            score = 1.0 - np.absolute(w_true - w_hat).mean()
            assert not np.isnan(score)
            scores.append(score)
        except: scores.append(float(0)) 
    return scores

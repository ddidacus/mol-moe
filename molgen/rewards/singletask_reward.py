import torch
from molgen.utils.constants import SPECIAL_CHARS
from molgen.utils.rlhf_toolkit import *
import torch.nn as nn

class SingleTaskRewardModel(nn.Module):
    def __init__(self, callable_reward_model, tokenizer):
        super().__init__()
        self.callable_reward_model = callable_reward_model
        self.tokenizer = tokenizer
        self.base_model_prefix = "prefix"
        self.prefix = self

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
                scores.append(self.callable_reward_model(text))
            except: scores.append(0)
        scores = torch.tensor(scores, device=hidden_states.device)
        return scores.unsqueeze(1).expand(-1, seq_len)  # Shape: (batch_size, seq_len)

    def modules(self):
        return []

    def to(self, _):
        return self

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import BertModel


class RecoBERTHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x: Tensor, token_types: Tensor, special_tokens: Tensor
    ) -> Tuple[Tensor, Tensor]:
        title_mask = (token_types == 0) & (special_tokens == 0)
        descr_mask = (token_types == 1) & (special_tokens == 0)

        t = x * title_mask.unsqueeze(-1)
        d = x * descr_mask.unsqueeze(-1)

        f_t = torch.sum(t, dim=1) / torch.sum(title_mask, dim=-1).unsqueeze(-1)
        f_d = torch.sum(d, dim=1) / torch.sum(title_mask, dim=-1).unsqueeze(-1)

        return f_t, f_d


class TitleDescriptionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, f_t: Tensor, f_d: Tensor) -> Tensor:
        cosine = (1.0 + self.cos_sim.forward(f_t, f_d)) / 2.0
        return cosine


class LanguageModelHead(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super().__init__()

        self.classifier = nn.Linear(in_features=input_dim, out_features=vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier.forward(x)


class RecoBERT(nn.Module):
    def __init__(self, bert: BertModel, vocab_size: int):
        super().__init__()

        self.bert = bert
        self.rb_head = RecoBERTHead()
        self.td_head = TitleDescriptionHead()
        self.lm_head = LanguageModelHead(bert.pooler.dense.out_features, vocab_size)

    def forward(
        self, input_ids: Tensor, attn_mask: Tensor, special_tokens: Tensor, token_types: Tensor
    ) -> Dict[str, Tensor]:
        y = self.bert.forward(input_ids=input_ids, attention_mask=attn_mask)
        features = y["last_hidden_state"]

        f_t, f_d = self.rb_head.forward(features, token_types, special_tokens)
        cos_sim = self.td_head.forward(f_t, f_d)
        lm_scores = self.lm_head.forward(features)

        return {"f_t": f_t, "f_d": f_d, "cos_sim": cos_sim, "lm_scores": lm_scores}

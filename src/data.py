import random
from typing import Any, Dict, List

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BertTokenizer, DataCollatorForWholeWordMask


class RecoDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        t_idx = idx
        d_idx = (
            random.choice([i for i in range(0, len(self.df)) if i != idx])
            if random.random() < 0.5
            else idx
        )
        label = int(t_idx == d_idx)
        title = self.df.iloc[t_idx]["title"]
        descr = self.df.iloc[d_idx]["description"]

        return {"title": title, "description": descr, "label": label}


class CollatorWrapper:
    def __init__(self, tokenizer: BertTokenizer, collator: DataCollatorForWholeWordMask):
        self.tokenizer = tokenizer
        self.collator = collator

    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        titles = [d["title"] for d in data]
        descrs = [d["description"] for d in data]
        encoded = self.tokenizer(
            titles,
            descrs,
            return_special_tokens_mask=True,
            padding=True,
            truncation=True,
            return_attention_mask=True,
        )
        masked = self.collator(encoded["input_ids"], return_tensors="pt")

        output = {
            "input_ids": masked["input_ids"],
            "token_type_ids": torch.tensor(encoded["token_type_ids"], dtype=torch.int32),
            "special_tokens_mask": torch.tensor(encoded["special_tokens_mask"], dtype=torch.int32),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.int32),
            "MLM_label": masked["labels"],
            "NSP_label": torch.tensor([d["label"] for d in data], dtype=torch.int32),
        }
        return output

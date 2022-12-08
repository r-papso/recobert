import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BertTokenizer, DataCollatorForWholeWordMask, RobertaTokenizer


class RecoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, swap_prob: float) -> None:
        super().__init__()

        self.df = df
        self.prob = swap_prob

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        t_idx = idx
        d_idx = (
            random.choice([i for i in range(0, len(self.df)) if i != idx])
            if random.random() < self.prob
            else idx
        )

        label = int(t_idx == d_idx)
        title = self.df.iloc[t_idx]["title"]
        descr = self.df.iloc[d_idx]["description"]
        t_id = self.df.iloc[t_idx]["id"]
        d_id = self.df.iloc[d_idx]["id"]

        return {"title": title, "description": descr, "label": label, "t_id": t_id, "d_id": d_id}


class CollatorBase(ABC):
    def __init__(self, tokenizer: Union[BertTokenizer, RobertaTokenizer]) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        pass

    def _encode(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        titles = [d["title"] for d in data]
        descrs = [d["description"] for d in data]
        encoded = self.tokenizer(
            titles,
            descrs,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            padding=True,
            truncation=True,
            return_attention_mask=True,
        )
        return encoded


class TrainCollator(CollatorBase):
    def __init__(
        self,
        tokenizer: Union[BertTokenizer, RobertaTokenizer],
        collator: DataCollatorForWholeWordMask,
    ) -> None:
        super().__init__(tokenizer)

        self.collator = collator

    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        encoded = self._encode(data)
        masked = self.collator(encoded["input_ids"], return_tensors="pt")

        output = {
            "input_ids": masked["input_ids"],
            "token_type_ids": torch.tensor(encoded["token_type_ids"], dtype=torch.int32),
            "special_tokens": torch.tensor(encoded["special_tokens_mask"], dtype=torch.int32),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.int32),
            "MLM_label": masked["labels"],
            "NSP_label": torch.tensor([d["label"] for d in data], dtype=torch.int32),
        }
        return output


class InferenceCollator(CollatorBase):
    def __init__(self, tokenizer: Union[BertTokenizer, RobertaTokenizer]) -> None:
        super().__init__(tokenizer)

    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        encoded = self._encode(data)

        output = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.int32),
            "token_type_ids": torch.tensor(encoded["token_type_ids"], dtype=torch.int32),
            "special_tokens": torch.tensor(encoded["special_tokens_mask"], dtype=torch.int32),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.int32),
            "title_id": torch.tensor([d["t_id"] for d in data], dtype=torch.int32),
            "descr_id": torch.tensor([d["d_id"] for d in data], dtype=torch.int32),
        }
        return output

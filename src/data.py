import random
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

        return {"title": title, "description": descr, "label": label}


class RobertaTokenizerWrapper:
    def __init__(self, tokenizer: RobertaTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(
        self,
        titles: List[str],
        descrs: List[str],
        return_special_tokens_mask: bool,
        return_token_type_ids: bool,
        padding: bool,
        truncation: bool,
        return_attention_mask: bool,
        return_tensors: str,
    ) -> Dict[str, Any]:
        encoded = self.tokenizer(
            titles,
            descrs,
            return_special_tokens_mask=return_special_tokens_mask,
            return_token_type_ids=return_token_type_ids,
            padding=padding,
            truncation=truncation,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
        )

        for i in range(len(encoded["input_ids"])):
            sep_count = 0

            for j in range(len(encoded["input_ids"][i])):
                if encoded["input_ids"][i][j] == self.tokenizer.sep_token_id:
                    sep_count += 1
                    continue

                if sep_count == 3:
                    break

                if sep_count == 2:
                    encoded["token_type_ids"][i][j] = 1

        return encoded


class CollatorWrapper:
    def __init__(
        self,
        tokenizer: Union[BertTokenizer, RobertaTokenizerWrapper],
        collator: DataCollatorForWholeWordMask,
    ) -> None:
        self.tokenizer = tokenizer
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

from typing import Dict, Tuple

import torch
from torch import Tensor


def train_batch_to(batch: Dict[str, Tensor], device: str) -> Tuple[Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    NSP_label = batch["NSP_label"].to(device).to(torch.float32)
    MLM_label = batch["MLM_label"].to(device)
    special_tokens_mask = batch["special_tokens"].to(device)
    token_type_ids = batch["token_types"].to(device)

    return input_ids, attention_mask, NSP_label, MLM_label, special_tokens_mask, token_type_ids


def infer_batch_to(batch: Dict[str, Tensor], device: str) -> Tuple[Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    special_tokens_mask = batch["special_tokens"].to(device)
    token_type_ids = batch["token_types"].to(device)
    title_id = batch["title_id"].to(device)
    descr_id = batch["descr_id"].to(device)

    return input_ids, attention_mask, special_tokens_mask, token_type_ids, title_id, descr_id

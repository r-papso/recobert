from typing import Dict, Tuple

import torch
from torch import Tensor


def batch_to(batch: Dict[str, Tensor], device: str) -> Tuple[Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    NSP_label = batch["NSP_label"].to(device).to(torch.float32)
    MLM_label = batch["MLM_label"].to(device)
    special_tokens = batch["special_tokens"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)

    return input_ids, attention_mask, NSP_label, MLM_label, special_tokens, token_type_ids

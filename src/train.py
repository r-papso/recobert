import os
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .model import RecoBERT


def batch_to(batch: Dict[str, Tensor], device: str) -> Tuple[Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    NSP_label = batch["NSP_label"].to(device).to(torch.float32)
    MLM_label = batch["MLM_label"].to(device)
    special_tokens_mask = batch["special_tokens_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)

    return input_ids, attention_mask, NSP_label, MLM_label, special_tokens_mask, token_type_ids


def evaluate(model: RecoBERT, val_loader: DataLoader, device: str) -> float:
    model = model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()
    be_loss_fn = nn.BCELoss()
    sum_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            in_ids, attn_mask, nsp_label, mlm_label, special_tokens, token_types = batch_to(
                batch, device
            )

            y = model.forward(
                input_ids=in_ids,
                attention_mask=attn_mask,
                special_tokens=special_tokens,
                token_types=token_types,
            )
            
            be_loss = be_loss_fn.forward(y["cos_sim"], nsp_label)
            ce_loss = ce_loss_fn.forward(y["lm_scores"].transpose(1, 2), mlm_label)
            loss = be_loss + ce_loss
            sum_loss += loss.detach().item()

    return sum_loss / len(val_loader)


def train(
    model: RecoBERT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optim: Optimizer,
    epochs: int,
    device: str,
    checkpoint: str,
    early_stop: int,
) -> RecoBERT:
    os.makedirs(checkpoint, exist_ok=True)

    model = model.train()
    ce_loss_fn = nn.CrossEntropyLoss()
    be_loss_fn = nn.BCELoss()
    best_loss, no_improve = 1e6, 0

    for epoch in range(epochs):
        sum_loss = 0

        for batch in train_loader:
            in_ids, attn_mask, nsp_label, mlm_label, special_tokens, token_types = batch_to(
                batch, device
            )

            optim.zero_grad()
            y = model.forward(
                input_ids=in_ids,
                attention_mask=attn_mask,
                special_tokens=special_tokens,
                token_types=token_types,
            )

            be_loss = be_loss_fn.forward(y["cos_sim"], nsp_label)
            ce_loss = ce_loss_fn.forward(y["lm_scores"].transpose(1, 2), mlm_label)
            loss = be_loss + ce_loss

            loss.backward()
            optim.step()
            sum_loss += loss.detach().item()

        time = datetime.now().strftime("%H:%M:%S")
        print(f"{time} - Epoch {(epoch):03d}: Train Loss = {(sum_loss / len(train_loader)):.4f}")

        val_loss = evaluate(model, val_loader, device)

        if val_loss < best_loss:
            fs = [f for f in os.listdir(checkpoint) if os.path.isfile(os.path.join(checkpoint, f))]
            _ = [os.remove(os.path.join(checkpoint, f)) for f in fs]

            best_loss = val_loss
            no_improve = 0
            torch.save(model, os.path.join(checkpoint, f"{epoch:03d}_{val_loss:.4f}.pth"))
        else:
            no_improve += 1

        time = datetime.now().strftime("%H:%M:%S")
        print(f"{time} - Epoch {epoch:03d}: Val Loss = {val_loss:.4f}")
        model = model.train()

        if no_improve >= early_stop:
            print(f"No improvement in {no_improve} epochs, early stopping...")
            break

    return model

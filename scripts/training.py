import argparse
import sys

import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import (
    BertModel,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    RobertaModel,
    RobertaTokenizer,
)

sys.path.append("..")

from src.data import CollatorWrapper, RecoDataset, RobertaTokenizerWrapper
from src.model import RecoBERT
from src.train import train

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.dataset == "wines":
        # Define hyper-parameters
        lr = 0.0001
        l2_reg = 0.0
        beta1 = 0.9
        beta2 = 0.999
        epochs = 100
        early_stop = 100
        batch_size = 32
        workers = 8

        # BERT (https://arxiv.org/pdf/1810.04805.pdf)
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        bert = BertModel.from_pretrained("bert-base-cased")
        collator = DataCollatorForLanguageModeling(tokenizer)
        collate_fn = CollatorWrapper(tokenizer, collator)

        # Load dataset
        df = pd.read_csv("../data/wines.csv", index_col=0)
        dataset = RecoDataset(df=df, swap_prob=0.5)

        # Dataset split
        idxs = list(range(len(dataset)))
        split = int(len(dataset) * 0.8)
        train_idxs, val_idxs = idxs[:split], idxs[split:]

        train_sampler = SubsetRandomSampler(train_idxs)
        val_sampler = SubsetRandomSampler(val_idxs)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=workers,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=workers,
        )

    elif args.dataset == "fashion":
        # Define hyper-parameters
        lr = 0.0001
        l2_reg = 0.00001
        beta1 = 0.9
        beta2 = 0.999
        epochs = 100
        early_stop = 100
        batch_size = 32
        workers = 8

        # SlovakBERT (https://arxiv.org/abs/2109.15254)
        tokenizer = RobertaTokenizer.from_pretrained("gerulata/slovakbert")
        tokenize_fn = RobertaTokenizerWrapper(tokenizer)
        bert = RobertaModel.from_pretrained("gerulata/slovakbert")
        collator = DataCollatorForLanguageModeling(tokenizer)
        collate_fn = CollatorWrapper(tokenize_fn, collator)

        # Load dataset
        df = pd.read_csv("../data/fashion.csv", index_col=0)
        dataset = RecoDataset(df=df, swap_prob=0.5)

        # Dataset split
        idxs = list(range(len(dataset)))
        trainval_split = int(len(dataset) * 0.6)
        valtest_split = int(len(dataset) * 0.8)
        train_idxs, val_idxs = idxs[:trainval_split], idxs[trainval_split:valtest_split]

        train_sampler = SubsetRandomSampler(train_idxs)
        val_sampler = SubsetRandomSampler(val_idxs)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=workers,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=workers,
        )

    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    model = RecoBERT(bert, tokenizer.vocab_size)

    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = model.to(device)

    optim = Adam(model.parameters(), lr=lr, weight_decay=l2_reg, betas=(beta1, beta2))

    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optim,
        epochs=epochs,
        device=device,
        checkpoint="./checkpoint",
        early_stop=early_stop,
    )

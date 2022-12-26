import argparse
import sys
from datetime import datetime

import pandas as pd
import torch
from transformers import BertTokenizer, RobertaTokenizer

sys.path.append("..")

from src.data import RobertaTokenizerWrapper
from src.metrics import MPR, MRR, HR_k
from src.model import TitleDescriptionHead

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str)


def eval_wines():
    # BERT (https://arxiv.org/pdf/1810.04805.pdf)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    df = pd.read_csv("../data/wines_annotated.csv", index_col=0)

    td_head = TitleDescriptionHead()
    model = torch.load("../scripts/checkpoint/097_0.7962.pth")

    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = model.eval().to(device)

    prev_seed = ""
    ratings, targets = [], []

    with torch.no_grad():
        for i, seed in df.iterrows():
            if seed["seed title"] == prev_seed:
                continue

            prev_seed = seed["seed title"]
            rating, target = [], []

            for j, reco in df.iterrows():
                seed_t, seed_d = seed["seed title"], seed["seed description"]
                reco_t, reco_d = reco["recommended title"], reco["recommended description"]

                tokens = tokenizer(
                    [seed_t, reco_t, seed_t, reco_t],
                    [seed_d, reco_d, reco_d, seed_d],
                    return_special_tokens_mask=True,
                    return_token_type_ids=True,
                    padding=True,
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                special_tokens = tokens["special_tokens_mask"].to(device)
                attn_mask = tokens["attention_mask"].to(device)
                input_ids = tokens["input_ids"].to(device)
                token_types = tokens["token_type_ids"].to(device)

                out = model.forward(
                    input_ids=input_ids,
                    attn_mask=attn_mask,
                    special_tokens=special_tokens,
                    token_types=token_types,
                )

                seed_ft, seed_fd = out["f_t"][0:1], out["f_d"][0:1]
                reco_ft, reco_fd = out["f_t"][1:2], out["f_d"][1:2]
                td_sim, dt_sim = out["cos_sim"][2:3], out["cos_sim"][3:4]

                tt_sim = td_head.forward(seed_ft, reco_ft)
                dd_sim = td_head.forward(seed_fd, reco_fd)

                total = sum([td_sim, dt_sim, tt_sim, dd_sim])
                label = seed["seed title"] == reco["seed title"]

                rating.append(total)
                target.append(label)

            ratings.append(rating)
            targets.append(target)

            if i > 0 and i % 10 == 0:
                print(f"Row {i} processed...")

    r = torch.tensor(ratings)
    y = torch.tensor(targets)

    HR10, HR50, HR100 = HR_k(10, r, y), HR_k(50, r, y), HR_k(100, r, y)
    _MRR, _MPR = MRR(r, y), MPR(r, y)

    prnt = f"HR@10 - {HR10}, HR@50 - {HR50}, HR@100 - {HR100}, MRR: {_MRR}, MPR: {_MPR}"
    print(f"Wine dataset evaluation: {prnt}")

    torch.save(r, "../data/wines_scores.pt")
    torch.save(y, "../data/wines_labels.pt")


def eval_fashion():
    # SlovakBERT (https://arxiv.org/abs/2109.15254)
    wrappee = RobertaTokenizer.from_pretrained("gerulata/slovakbert")
    tokenizer = RobertaTokenizerWrapper(wrappee)

    df = pd.read_csv("../data/fashion.csv", index_col=0)
    split = int(len(df) * 0.8)
    test_df = df[split:]

    td_head = TitleDescriptionHead()
    model = torch.load("../scripts/models/fashion_095_0.5390.pth")

    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = model.eval().to(device)

    ratings, targets = [], []

    with torch.no_grad():
        for i, seed in test_df.iterrows():
            if len(test_df[test_df["label"] == seed["label"]]) <= 1:
                continue

            rating, target = [], []

            for j, reco in test_df.iterrows():
                if seed["id"] == reco["id"]:
                    continue

                seed_t, seed_d = seed["title"], seed["description"]
                reco_t, reco_d = reco["title"], reco["description"]

                tokens = tokenizer(
                    [seed_t, reco_t, seed_t, reco_t],
                    [seed_d, reco_d, reco_d, seed_d],
                    return_special_tokens_mask=True,
                    return_token_type_ids=True,
                    padding=True,
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                special_tokens = tokens["special_tokens_mask"].to(device)
                attn_mask = tokens["attention_mask"].to(device)
                input_ids = tokens["input_ids"].to(device)
                token_types = tokens["token_type_ids"].to(device)

                out = model.forward(
                    input_ids=input_ids,
                    attn_mask=attn_mask,
                    special_tokens=special_tokens,
                    token_types=token_types,
                )

                seed_ft, seed_fd = out["f_t"][0:1], out["f_d"][0:1]
                reco_ft, reco_fd = out["f_t"][1:2], out["f_d"][1:2]
                td_sim, dt_sim = out["cos_sim"][2:3], out["cos_sim"][3:4]

                tt_sim = td_head.forward(seed_ft, reco_ft)
                dd_sim = td_head.forward(seed_fd, reco_fd)

                total = sum([td_sim, dt_sim, tt_sim, dd_sim])
                label = seed["label"] == reco["label"]

                rating.append(total)
                target.append(label)

            ratings.append(rating)
            targets.append(target)

            if i > 0 and i % 10 == 0:
                time = datetime.now().strftime("%H:%M:%S")
                print(f"{time}: Row {i} processed...")

    r = torch.tensor(ratings)
    y = torch.tensor(targets)

    HR10, HR50, HR100 = HR_k(10, r, y), HR_k(50, r, y), HR_k(100, r, y)
    _MRR, _MPR = MRR(r, y), MPR(r, y)

    prnt = f"HR@10 - {HR10}, HR@50 - {HR50}, HR@100 - {HR100}, MRR: {_MRR}, MPR: {_MPR}"
    print(f"Fashion dataset evaluation: {prnt}")

    torch.save(r, "../data/fashion_scores.pt")
    torch.save(y, "../data/fashion_labels.pt")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.dataset == "wines":
        eval_wines()
    elif args.dataset == "fashion":
        eval_fashion()
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

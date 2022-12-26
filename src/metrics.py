import torch
from torch import Tensor


def HR_k(k: int, r: Tensor, y: Tensor) -> float:
    r_sort, idxs = torch.sort(r, descending=True)
    y_sort = torch.gather(y, dim=1, index=idxs)
    total = 0

    for i in range(y_sort.shape[0]):
        total += torch.sum(y_sort[i][:k]) > 0

    return total / y_sort.shape[0]


def MRR(r: Tensor, y: Tensor) -> float:
    r_sort, idxs = torch.sort(r, descending=True)
    y_sort = torch.gather(y, dim=1, index=idxs)
    RR = 0

    for i in range(y_sort.shape[0]):
        rank = torch.nonzero(y_sort[i])[0].item() + 1
        RR += 1.0 / rank

    return RR / y_sort.shape[0]


def MPR(r: Tensor, y: Tensor) -> float:
    r_sort, idxs = torch.sort(r, descending=True)
    y_sort = torch.gather(y, dim=1, index=idxs)
    PR = 0

    for i in range(y_sort.shape[0]):
        ranks = torch.nonzero(y_sort[i])
        percentiles = (ranks.squeeze() + 1) / y_sort[1].shape[0]
        PR += torch.mean(percentiles).item()

    return PR / y_sort.shape[0]

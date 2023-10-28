from typing import List

import torch
from sklearn.metrics import f1_score


def calc_accuracy(pred: List[List], target: List[List]) -> float:
    summ = sum([1 for yhat, y in zip(pred, target) if yhat == y]) / len(target)
    return summ


def calc_f1(pred: List[List], target: List[List]) -> float:
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy()
    if type(target) == torch.Tensor:
        target = target.detach().cpu().numpy()
    score = f1_score(pred, target, average="macro")
    return score

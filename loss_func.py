import torch
import torch.nn as nn
import torch.nn.functional as F


class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, custom_loss: float, alpha: float) -> float:
        nll = F.nll_loss(predictions, targets)
        loss = ((alpha * nll) + ((1 - alpha) * custom_loss))
        return loss

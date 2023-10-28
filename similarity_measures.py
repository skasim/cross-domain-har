import logging
from typing import List

import torch
import torch.nn as nn
from cca_zoo.models import TCCA, KTCCA
import numpy as np
import scipy

def MMD(x: torch.Tensor, y: torch.Tensor, sigma: int, device: str) -> torch.Tensor:
    # https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
    """Empirical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        sigma: sigma value
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    XX += torch.exp(-0.5 * dxx / sigma)
    YY += torch.exp(-0.5 * dyy / sigma)
    XY += torch.exp(-0.5 * dxy / sigma)

    mmd = torch.mean(XX + YY - 2. * XY)
    return mmd


def cosine_similarity(mat1: torch.Tensor, mat2: torch.Tensor)-> float:
    cos = nn.CosineSimilarity(dim=1)
    sim = cos(mat1, mat2)
    return 1 - torch.mean(sim).detach().cpu().numpy()


def cca_similarity_loss(tensor1: torch.Tensor, tensor2: torch.tensor, is_kcca: bool=False)-> float:
    torch.nan_to_num(tensor1)
    torch.nan_to_num(tensor2)
    mat1 = tensor1.detach().cpu().numpy()
    mat2 = tensor2.detach().cpu().numpy()
    if is_kcca:
        model = KTCCA(kernel="rbf")
    else:
        model = TCCA()
    try:
        model.fit((mat1, mat2))
    except ValueError:
        mat1 = np.nan_to_num(mat1)
        mat2 = np.nan_to_num(mat2)
        model.fit((mat1, mat2))
    except (np.linalg.LinAlgError, Exception) as e:
        logging.error(f"error encountered at cca model.fit: {e}")
        return .5
    return 1 - model.score((mat1, mat2))[0]


def euclidean_distance(mat1: torch.Tensor, mat2: torch.Tensor)->float:
    return torch.dist(mat1, mat2, p=2).detach().numpy()


import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']

@LOSS_REGISTRY.register()
class FidelityLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(FidelityLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = 1e-8
    def forward(self, p, g):
        p = p.view(-1)
        g = g.view(-1)
        loss = 1 - (torch.sqrt(p*g + self.eps) + torch.sqrt((1-p)*(1-g) + self.eps))
        return torch.mean(self.loss_weight*loss)

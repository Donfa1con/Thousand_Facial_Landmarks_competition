import torch
import torch.nn as nn


class CustomL2Loss(nn.Module):
    def forward(self, y_true, y_pred):
        return torch.sqrt(torch.pow(y_true - y_pred, 2).reshape(len(y_true), -1, 2).sum(axis=2)).mean(axis=1).sum()

import torch
import torch.nn as nn
import torch.nn.functional as F


def cone(x: torch.Tensor) -> torch.Tensor:
    return 1 - torch.abs(x - 1)


def parabolic_cone(x: torch.Tensor) -> torch.Tensor:
    return x * (2 - x)


def parameterized_cone(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return 1 - torch.abs(x - 1).pow(beta)


class ParameterizedCone(nn.Module):

    def __init__(self, beta: float = 1.) -> torch.Tensor:
        self.beta = nn.Parameter(beta, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return parameterized_cone(x, self.beta)


class ParabolicCone(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return parabolic_cone(x)


class Cone(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cone(x)

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

    def __init__(self, beta: torch.Tensor | None) -> torch.Tensor:
        self.beta = beta if beta is not None else nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return parameterized_cone(x, self.beta)


class ParabolicCone(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return parabolic_cone(x)


class Cone(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cone(x)

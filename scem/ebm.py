"""Module containing implementations of 
energy-based (unnormalised density) models
"""

import torch
import torch.nn as nn
from scem import stein
from abc import ABCMeta, abstractmethod
from torch.nn.parameter import Parameter


class LatentEBM(nn.Module, metaclass=ABCMeta):
    """Base class for energy-based
    (or unnormalized density) models with
    a latent variables.

    Attributes:
        var_type_latent: indicates the type of the
            latent variable.
    """
    def __init__(self):
        super(LatentEBM, self).__init__()

    @abstractmethod
    def forward(self, X, Z):
        """Computes log unnormalised density

        Args:
            X (torch.Tensor): tensor of size [batch_size, ...]
            Z (torch.Tensor): tensor of size [batch_size, ...]
        Returns:
            torch.Tensor: tensor of size [batch_size, ...]
        """
        pass

    def energy_grad_obs(self, X, Z):
        """Computes the gradient
        of the energy function
        (unnormalised part) w.r.t. X


        Args:
            X (torch.Tensor): tensor of size [n, ...]
            Z (torch.Tensor): tensor of size [n, ...]

        Returns:
            torch.Tensor: tensor of size [n, ...]
        """
        assert isinstance(X, torch.Tensor)
        assert isinstance(Z, torch.Tensor)
        X.requires_grad = True
        energy_sum = torch.sum(self.forward(X, Z))
        Gs = torch.autograd.grad(
            energy_sum, X, retain_graph=True, only_inputs=True)
        G = Gs[0]

        n, dx = X.shape
        assert G.shape[0] == n
        assert G.shape[1] == dx
        return G

    def posterior_score(self, X, Z):
        var_type = self.var_type_latent
        score_fn = stein.pscore_dict[var_type]
        return score_fn(X, Z, self)

    @property
    @abstractmethod
    def var_type_latent(self):
        """Returns the type of the latent variable"""
        pass

    def score_marginal_obs(self, X):
        return None

    def has_score_marginal_obs(self, X):
        return self.score_marginal_obs() is not None


class PPCA(LatentEBM):
    """A class representing a PPCA model.

    Attributes:
        weight:
            torch.Tensor representing a weight matrix
            in the observation likelihood. The size is
            dx x dz.
        var:
            torch.Tensor representing a the variatnce
            of the observation likelihood. A scalar.
    """
    var_type_latent = 'continuous'

    def __init__(self, weight, var):
        super(PPCA, self).__init__()
        self.weight = Parameter(weight)
        self.var = Parameter(var)

    def forward(self, X, Z):
        W = self.weight
        var = self.var
        mean = Z @ W.T
        X0 = X - mean
        exponent = -0.5/var * torch.sum(X0**2, dim=-1)
        exponent += -0.5 * torch.sum(Z**2, dim=-1)
        return exponent

    def score_marginal_obs(self, X):
        n, dx = X.shape
        W = self.weight
        var = self.var
        cov = var*torch.eye(dx) + W@W.T
        return - X @ torch.pinverse(cov)

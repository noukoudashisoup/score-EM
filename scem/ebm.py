"""Module containing implementations of 
energy-based (unnormalised density) models
"""

import torch
import torch.nn as nn
from scem import util
from abc import ABCMeta, abstractmethod
from torch.nn.parameter import Parameter


class DiscreteModel(metaclass=ABCMeta):
    """Interface for models with
    variables defined on a lattice
    """

    @abstractmethod
    def lattice_ranges(self, key):
        """Returns lattice size 
        for the key = {'obs', 'latent'}

        Args:
            key (str): 
                str expected to take a value in
                {'obs', 'latent'}
        """
        pass


class LatentEBM(nn.Module, metaclass=ABCMeta):
    """Base class for energy-based
    (or unnormalized density) models with
    a latent variables.

    Attributes:
        var_type_obs: indicates the type of the
            observable variable
        var_type_latent: indicates the type of the
            latent variable.
    """
    def __init__(self):
        super(LatentEBM, self).__init__()

    @property
    @abstractmethod
    def var_type_obs(self):
        """Returns the type of the observable variable"""
        pass

    @property
    @abstractmethod
    def var_type_latent(self):
        """Returns the type of the latent variable"""
        pass

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

    def score_joint_obs(self, X, Z):
        """Computes the gradient
        of the energy function
        (unnormalised part) w.r.t. X. 
        Subclasses may extend this if it is more efficient.


        Args:
            X (torch.Tensor): tensor of size [n, dx] 
            Z (torch.Tensor): tensor of size [n, dz] 

        Returns:
            torch.Tensor: tensor of size [n, dx] 
        """
        assert isinstance(X, torch.Tensor)
        assert isinstance(Z, torch.Tensor)
        var_type = self.var_type_obs
        score_fn = ebm_oscore_dict[var_type]
        return score_fn(X, Z, self)

    def score_joint_latent(self, X, Z):
        """Computes the gradient
        of the energy function
        (unnormalised part) w.r.t. Z
        Subclasses may extend this if it is more efficient.


        Args:
            X (torch.Tensor): tensor of size [n, dx] 
            Z (torch.Tensor): tensor of size [n, dz]

        Returns:
            torch.Tensor: tensor of size [n, dz]
        """

        var_type = self.var_type_latent
        score_fn = ebm_lscore_dict[var_type]
        return score_fn(X, Z, self)

    def score_marginal_obs(self, X):
        """Returns the score of p(x) = \int p(x,z)dz

        Args:
            X (torch.Tensor): tensor of size [n, dx]

        Returns:
            [torch.Tensor]: tensor of size [n, dx]
        """
        return None

    def has_score_marginal_obs(self, X):
        return self.score_marginal_obs() is not None


def score_latent_cont(X, Z, ebm):
    """Computes the derivative of ebm.forward(x,z)
    w.r.t z

    Args:
        X (torch.Tensor):
            torch tensor of size [n, dx] 
        Z (torch.Tensor):
            torch tensor of size [n, dz]

        ebm (LatentEBM):
            latentEBM object

    Returns:
        [torch.Tensor]:
            tensor of size [n, dz]
            The derivative of func w.r.t. Z
            evaluated at X, Z. 
    """
    assert isinstance(X, torch.Tensor)
    assert isinstance(Z, torch.Tensor)
    return util.gradient(ebm.forward, 1, [X, Z])


def score_obs_cont(X, Z, ebm):
    """Computes the derivative of ebm.forward(x,z)
    w.r.t X

    Args:
        X (torch.Tensor):
            torch tensor of size [n, dx] 
        Z (torch.Tensor):
            torch tensor of size [n, dz]

        ebm (LatentEBM):
            latentEBM object

    Returns:
        [torch.Tensor]:
            tensor of size [n, dz]
            The derivative of func w.r.t. X
            evaluated at X, Z. 
    """
    assert isinstance(X, torch.Tensor)
    assert isinstance(Z, torch.Tensor)
    return util.gradient(ebm.forward, 0, [X, Z])


def score_obs_lattice(X, Z, ebm):
    assert isinstance(X, torch.Tensor)
    assert isinstance(Z, torch.Tensor)
    which_var = 'obs'
    ls = ebm.lattice_ranges(which_var)
    D = util.forward_diff(ebm.forward, 0,
                          [X, Z], ls)
    return torch.exp(D) - 1.


def score_latent_lattice(X, Z, ebm):
    assert isinstance(X, torch.Tensor)
    assert isinstance(Z, torch.Tensor)
    which_var = 'latent'
    ls = ebm.lattice_ranges(which_var)
    D = util.forward_diff(ebm.forward, 1,
                          [X, Z], ls)
    return torch.exp(D) - 1.


def score_latent_lattice_onehot(X, Z, ebm):
    assert isinstance(X, torch.Tensor)
    assert isinstance(Z, torch.Tensor)
    D = util.forward_diff_onehot(ebm.forward,
                                 1, [X, Z])
    return torch.exp(D) - 1.


def score_obs_lattice_onehot(X, Z, ebm):
    assert isinstance(X, torch.Tensor)
    assert isinstance(Z, torch.Tensor)
    D = util.forward_diff_onehot(ebm.forward,
                                 0, [X, Z])
    return torch.exp(D) - 1.


# Dictionary of joint score functions
# with derivatives computed w.r.t. X
ebm_oscore_dict = {
    'continuous': score_obs_cont,
    'lattice': score_obs_lattice,
    'lattice_onehot': score_latent_lattice_onehot,
}
# Dictionary of posterior score functions
ebm_lscore_dict = {
    'continuous': score_latent_cont,
    'lattice': score_latent_lattice,
    'lattice_onehot': score_latent_lattice_onehot,
}


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
    var_type_obs = 'continuous'
    var_type_latent = 'continuous'

    def __init__(self, weight, var):
        super(PPCA, self).__init__()
        self.weight = Parameter(weight)
        self.var = Parameter(var)

    def forward(self, X, Z):
        W = self.weight
        std = self.var**0.5
        mean = Z @ W.T
        X0 = X - mean
        exponent = -0.5/std**2 * torch.sum(X0**2, dim=-1)
        exponent += -0.5 * torch.sum(Z**2, dim=-1)
        return exponent

    def score_marginal_obs(self, X):
        n, dx = X.shape
        W = self.weight
        std = self.var**0.5
        cov = std**2 * torch.eye(dx) + W@W.T
        return - X @ torch.pinverse(cov)


class GaussianRBM(LatentEBM):
    """Class representing Gaussian Restricted Boltzmann
    Machines (GRBMs).

    \log p(x,z) = -x^T W z -b^Tx -c^Tz-||x||^2

    Attritbutes:
        W:
            weight matrix, initlaised at
            the given value
        b:
            coefficent vector for the observable
            initlaised at the given value
        c:
            coefficient vector for the latent,
            initlaised at the given value
    """
    var_type_obs = 'continuous'
    var_type_latent = 'lattice_onehot'

    def __init__(self, W, b, c):
        super(GaussianRBM, self).__init__()
        self.W = Parameter(W)
        self.b = Parameter(b)
        self.c = Parameter(c)
        self.dx = W.shape[0]
        self.dz = W.shape[1]
        ls_latent_ = 2*torch.ones([self.dz],
                                  dtype=torch.int)
        self.lattice_ranges_ = {'latent': ls_latent_}

    def forward(self, X, Z):
        Z_ = torch.sum(Z * torch.arange(2), dim=2)
        W = self.W
        b = self.b
        c = self.c
        log_joint = -torch.sum((X@W+c) * Z_, dim=1)
        log_joint += -X@b - torch.sum(X**2, axis=1)
        return log_joint

    def lattice_ranges(self, key):
        return self.lattice_ranges_[key]

    def score_marginal_obs(self, X):
        S = (-2.*X - self.b)
        a = -(X @ self.W + self.c)
        v = torch.sigmoid(a)
        S -= torch.einsum('ij, jl->il', v, self.W.T)
        return S

    def sample(self, n, seed=93, n_burnin=500, return_latent=False):
        W = self.W
        b = self.b
        c = self.c
        dx = self.dx
        dz = self.dz

        def gibbs(X0, n_sample, keep_sample=False):
            if keep_sample:
                X_batch = torch.empty([n_sample, dx])
                Z_batch = torch.empty([n_sample, dz])
            X_ = X0.clone()
            for i in range(n_sample):
                # Z|X
                pzx = torch.sigmoid(-(X_@W+c))
                Z_ = torch.bernoulli(pzx)
                # X|Z
                X_ = torch.randn(dx)/(2**0.5) - 0.5*(Z_@W.T + b)
                if keep_sample:
                    X_batch[i] = X_.detach()
                    Z_batch[i] = Z_.detach()
            if keep_sample:
                return X_batch, Z_batch
            return X_.detach(), Z_.detach()

        with util.TorchSeedContext(seed):
            X0 = torch.randn(dx)
            X_, _ = gibbs(X0, n_burnin)
            X, Z = gibbs(X_, n, keep_sample=True)
        if return_latent:
            return X, Z
        return X


class LatentEBMAdapter(LatentEBM):
    """Construct a LatentEBM object
    with its forward function specified
    by torch.nn.Module.

    Attributes:
        module: 
            a torch.nn.Module object
    """

    def __init__(self, module, var_type_obs, var_type_latent):
        super(LatentEBM, self).__init__()
        self.module = module
        self._var_type_obs = var_type_obs
        self._var_type_latent = var_type_latent

    def forward(self, X, Z):
        return self.module.forward(X, Z)
    
    @property
    def var_type_obs(self):
        return self._var_type_obs

    @property
    def var_type_latent(self):
        return self._var_type_latent
    


def main():
    seed = 13
    torch.manual_seed(seed)
    n = 13
    dx = 4
    dz = 2
    X = torch.randn([n, dx])
    Z = torch.ones([n, dz])
    W = torch.randn([dx, dz])
    var = torch.tensor([2.0])
    ppca = PPCA(W, var)
    s1 = ppca.score_joint_latent(X, Z)
    s2 = ppca.score_joint_obs(X, Z)


if __name__ == '__main__':
    main()
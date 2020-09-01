import torch
import torch.nn.functional as F
from scem import stein
from scem import util
from abc import ABCMeta, abstractmethod
from torch.nn.parameter import Parameter


class LatentEBM(torch.nn.Module, metaclass=ABCMeta):
    """Base class for energy-based
    (or unnormalized density) models

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

        """
        pass

    def energy_grad_obs(self, X, Z):
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


class ConditionalSampler(metaclass=ABCMeta):
    """Abstract class of conditional distributions"""

    @abstractmethod
    def sample(self, n_sample, *input,
               seed=3):
        """Conditioned on the input, generate
        n_sample samples.
        
        This class represents a conditinal
        distribution. Subclasses should
        implement samplers such that
        given an input, they output
        tensor of (n_sample,) + input.shape.
        """
        pass


class CSNoiseTransformer(ConditionalSampler,
                         torch.nn.Module):

    def __init__(self):
        super(CSNoiseTransformer, self).__init__()

    @abstractmethod
    def forward(self, noise, *input):
        """Map transforming noise
        """
        pass

    @abstractmethod
    def sample_noise(self, n_sample, *input, seed=13):
        """Sample from the noise distribution"""
        pass


class PTPPCAPosterior(ConditionalSampler):
    """Pytorch implementation of PPCA
    posterior.

    Attributes:
        ppca:   
            PPCA object
    """

    def __init__(self, ppca):
        super(PTPPCAPosterior, self).__init__()
        self.ppca = ppca

    def sample(self, n_sample, X,
               seed=3,
               ):
        with util.TorchSeedContext(seed):
            n = X.shape[0]
            W = self.ppca.weight
            _, dz = W.shape
            var = self.ppca.var
            cov = torch.pinverse(
                torch.eye(dz) + (W.T @ W)/var)
            mean = (X@W)@cov / var
            Z = (mean + torch.randn([n_sample, n, dz]) @ cov)
        return Z
    

class PTCSGaussLinearMean(CSNoiseTransformer):
    
    def __init__(self, dx, dz):
        super(PTCSGaussLinearMean, self).__init__()
        self.mean_fn = torch.nn.Linear(dx, dz)
        #self.cov = Parameter(0.01*torch.randn(dz, dz)) + torch.eye(dz)
        self.cov = Parameter(torch.eye(dz))
        self.dx = dx
        self.dz = dz
    
    def forward(self, noise, X):
        mean = (self.mean_fn(X))
        cov = self.cov
        #print(cov @ cov.T)
        out  = noise @ cov  + mean
        return out

    def sample_noise(self, n_sample, X, seed=13):
        n = X.shape[0]
        return torch.randn(n_sample, n, self.dz)

    def sample(self, n_sample, X, seed=3):
        with util.TorchSeedContext(seed):
            noise = self.sample_noise(n_sample, X)
        return self.forward(noise, X)


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

    s = ppca.posterior_score(X, Z)
    ppca_post_score = -(Z@W.T@W-X@W)/var - Z
    #cs = PTPPCAPosterior(ppca)
    cs.apply(init_weights)
    post_score_mse = torch.mean((s-ppca_post_score)**2)
    print('Posterior score mse: {}'.format(post_score_mse))

    n_sample = 800
    assert isinstance(ppca, PPCA)
    approx_score = stein.ApproximateScore(
        ppca.energy_grad_obs, cs)
    marginal_score_mse = (torch.mean(
        (approx_score(X, n_sample=n_sample)-ppca.score_marginal_obs(X))**2))
    print('Marginal score mse: {}'.format(marginal_score_mse))


if __name__ == '__main__':
    main()

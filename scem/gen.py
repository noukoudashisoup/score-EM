"""Module for generative models"""

import torch
import torch.nn as nn
import torch.distributions as dists
from scem import stein
from scem import util
from abc import ABCMeta, abstractmethod
from torch.nn.parameter import Parameter


class ConditionalSampler(metaclass=ABCMeta):
    """Abstract class of conditional distributions"""

    @abstractmethod
    def sample(self, n_sample, X, 
               seed=3, *args, **kwargs):
        """Conditioned on the input X, generate
        n_sample samples.
        
        This class represents a conditinal
        distribution. Subclasses should
        implement samplers such that
        given an input, they output
        tensor of (n_sample,) + X.shape[0].
        """
        pass


class CSNoiseTransformer(ConditionalSampler,
                         nn.Module):

    def __init__(self):
        super(CSNoiseTransformer, self).__init__()

    @abstractmethod
    def forward(self, noise, X, *args, **kwargs):
        """Map transforming noise
        """
        pass

    @abstractmethod
    def sample_noise(self, n_sample, n, seed=13):
        """Sample from the noise distribution

        Returns:
            torch.Tensor of sizee [n_sample, n, in_shape] 
        """
        pass

    @abstractmethod
    def in_out_shapes(self):
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
    """Gaussian distribution    of the form
    N(m(x), W W^T) where
    m(x) is an affine transformation and 
    W is a some matrix of dz x dz. 

    Attributes: 
        dx (int): dimensionality of the observable
        variable
        dz (int): dimensionality of the latent
        mean_fn(nn.Module):
            mean function, nn.Linear
        W: torch parameter, matrix of size [dz, dz]
    """
    
    def __init__(self, dx, dz):
        super(PTCSGaussLinearMean, self).__init__()
        self.mean_fn = nn.Linear(dx, dz)
        self.W = Parameter(torch.eye(dz))
        self.dx = dx
        self.dz = dz
    
    def forward(self, noise, X):
        W = self.W
        mean = self.mean_fn(X)
        out = noise @ W + mean
        return out

    def sample_noise(self, n_sample, X, seed=13):
        n = X.shape[0]
        return torch.randn(n_sample, n, self.dz)

    def sample(self, n_sample, X, seed=3):
        with util.TorchSeedContext(seed):
            noise = self.sample_noise(n_sample, X)
        return self.forward(noise, X)

    def in_out_shapes(self):
        return (self.dz, self.dz)


class CSTwoLayerGRBMQ(ConditionalSampler,
                                 nn.Module):
    
    n_cat = 2

    def __init__(self, grbm, n_particles=10):
        super(CSTwoLayerGRBMQ, self).__init__()
        self.grbm = grbm
        dx, dz = grbm.W.shape
        d1 = 64
        d2 = 64
        self.feat = nn.Sequential(
            nn.Linear(dx, d1),
            nn.ReLU(),
            nn.Linear(d1, d2),
            nn.ReLU(),
        )
        self.probs = [
            nn.Sequential(
                nn.Linear(d2, 2),
                nn.Softmax(dim=-1),
            )
            for _ in range(dz)
        ]

    def forward(self, X):
        feat = self.feat(X)
        out = [f(feat) 
               for f in self.probs]
        out = torch.stack(out, dim=0)
        return out

    def sample(self, n_sample, X,
               seed=3, *args, **kwargs):
        probs = self.forward(X)
        temp = torch.tensor([0.5])
        if self.training:
            m = dists.RelaxedOneHotCategorical(
                temp,
                probs,
            )
        else:
            m = dists.OneHotCategorical(probs)
        return m.sample([n_sample]).permute(0, 2, 1, 3)


def main():
    from scem.ebm import PPCA
    seed = 13
    torch.manual_seed(seed)
    n = 200
    dx = 4
    dz = 2
    X = torch.randn([n, dx])
    Z = torch.ones([n, dz])
    W = torch.randn([dx, dz])
    var = torch.tensor([10.0])
    ppca = PPCA(W, var)

    s = ppca.score_joint_latent(X, Z)
    ppca_post_score = -(Z@W.T@W-X@W)/var - Z
    cs = PTPPCAPosterior(ppca)
    # cs.apply(init_weights)
    post_score_mse = torch.mean((s-ppca_post_score)**2)
    print('Posterior score mse: {}'.format(post_score_mse))

    n_sample = 300
    assert isinstance(ppca, PPCA)
    approx_score = stein.ApproximateScore(
        ppca.score_joint_obs, cs)
    marginal_score_mse = (torch.mean(
        (approx_score(X, n_sample=n_sample)-ppca.score_marginal_obs(X))**2))
    print('Marginal score mse: {}'.format(marginal_score_mse))


if __name__ == '__main__':
    main()

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

    def log_den(self, *args, **kwargs):
        pass
        

class CSNoiseTransformer(ConditionalSampler,
                         nn.Module):
    """Conditional distribution of the form 
      Z \sim F(X, n) where X is a conditinoning
      variable, n is noise, and F is a function
      of those.  
    """

    def __init__(self):
        super(CSNoiseTransformer, self).__init__()

    @abstractmethod
    def forward(self, noise, X, *args, **kwargs):
        """Define map F transforming noise and input X 
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
        """Returns the tuple of the
        respective shapes of the noise 
        and the tranformed noise.
        """
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

        W = ppca.weight
        _, dz = W.shape
        var = ppca.var
        cov = torch.pinverse(
            torch.eye(dz) + (W.T @ W)/var)
        self.cov = cov

    def _mean_cov(self, X):
        var = self.ppca.var
        cov = self.cov
        mean = (X@W)@cov / var
        return mean, cov

    def sample(self, n_sample, X,
               seed=3,
               ):
        n = X.shape[0]
        W = self.ppca.weight
        _, dz = W.shape
        mean, cov = self._mean_cov(X)
        with util.TorchSeedContext(seed):
            Z = (mean + torch.randn([n_sample, n, dz]) @ cov)
        return Z
    
    


class PTCSGaussLinearMean(CSNoiseTransformer):
    """Gaussian distribution    of the form
    N(Ax+b, W W^T) where
    W is a some matrix of dz x dz. 

    Attributes: 
        dx (int): dimensionality of the observable
        variable
        dz (int): dimensionality of the latent
        mean_fn(nn.Module):
            mean function, nn.Linear
        W: torch parameter, matrix of size [dz, dz]
    """
    
    def __init__(self, dx, dz, *args, **kwargs):
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


class CSGRBMBernoulliFamily(ConditionalSampler,
                            nn.Module):
    """Class representing a conditional distribution
    of the form:
        \prod_{j=1}^dz Bern(z_j; pj(X)), where 
        pj(X) = softmax(AjX + bj)

    Attributes: 
        dx (int): 
            Conditioning variable's dimension
        dz (int):
            Output variables's dimension
    """
    n_cat = 2

    def __init__(self, dx, dz):
        super(CSGRBMBernoulliFamily, self).__init__()
        self.dx = dx
        self.dz = dz
        self.probs = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(dx, self.n_cat),
                nn.Softmax(dim=-1),
            )
             for _ in range(dz)
             ]
        )

    def forward(self, X):
        out = [f(X) 
               for f in self.probs]
        out = torch.stack(out, dim=0)
        return out

    def sample(self, n_sample, X,
               seed=3, *args, **kwargs):
        """
        Returns:
            torch.Tensor: tensor of size 
            [n_sample,] + X.shape + [2,]
        """
        probs = self.forward(X)
        temp = torch.tensor([1.], dtype=X.dtype)
        if self.training:
            m = dists.RelaxedOneHotCategorical(
                temp,
                probs,
            )
            return m.rsample([n_sample]).permute(0, 2, 1, 3)
        else:
            m = dists.OneHotCategorical(probs)
            return m.sample([n_sample]).permute(0, 2, 1, 3)


class CSGRBMPosterior(ConditionalSampler):
    """The posterior distribution of a Gaussian-Boltzmann
    Machine.

    Attributes: 
        grbm (ebm.GRBM)
        W (torch.Tensor):
            W parameter of grbm
        b (torch.Tensor)
            b paramter of grbm
        c (torch.Tensor)
            c parameter of grbm
    """

    def __init__(self, grbm):
        self.grbm = grbm
        self.W = grbm.W
        self.b = grbm.b
        self.c = grbm.c

    def sample(self, n_sample, X, seed=13):
        """
        Returns:
            torch.Tensor: tensor of size 
            [n_sample,] + X.shape + [2,]
        """

        W = self.W
        c = self.c
        probs = torch.sigmoid(-(X@W+c))
        probs = torch.stack([1.-probs, probs], dim=2)
        m = dists.OneHotCategorical(probs)
        with util.TorchSeedContext(seed):
            H = m.sample([n_sample])
        return H


class CSFactorisedGaussian(ConditionalSampler, nn.Module):

    def __init__(self, dx, dz, dh):
        super(CSFactorisedGaussian, self).__init__()
        self.dx = dx
        self.dz = dz
        self.dh = dh 
        self.layer_1 = nn.Linear(dx, dh)
        self.layer_2_m = nn.Linear(dh, dz)
        self.layer_2_v = nn.Linear(dh, dz)       

    def forward(self, X):
        h = self.layer_1(X).relu()
        m = self.layer_2_m(h)
        v = self.layer_2_v(h)
        v = nn.functional.softplus(v)
        return m, v
    
    def sample(self, n_sample, X, seed=3):
        n = X.shape[0]
        m, v = self.forward(X)
        d = m.shape[1]
        with util.TorchSeedContext(seed):
            noise = torch.randn(n_sample, n, d)
        return v * noise + m


class CSNoiseTransformerAdapter(CSNoiseTransformer):
    """Construct a CSNoiseTransformer having a given 
    torch.nn.Module as the transformation function.
     
    Attributes:
        - module (torch.nn.Module):
            A module serves as  a forward function.
            Assume that it has arguments f(X, noise) 
        - noise_sampler:
            noise sampler
        - in_out_shapes:
            tuple of the input and output shapes of noise
        - tensor_type:
            defines a tensor type of the noise 
        
    """

    def __init__(self, module, noise_sampler, in_out_shapes, tensor_type=torch.cuda.FloatTensor):
        super(CSNoiseTransformerAdapter, self).__init__()
        self.module = module
        self.noise_sampler = noise_sampler
        self.in_out_shapes = in_out_shapes
        self.tensor_type = tensor_type

    def forward(self, noise, X, *args, **kwargs):
        return self.module.forward(noise, X, *args, **kwargs)

    def sample_noise(self, n_sample, n, seed=13):
        """Returns (n_sample, n,)+in_out_shape[0] tensor"""
        tt = self.tensor_type
        noise = self.noise_sampler(n_sample, n, seed).type(tt)
        return noise

    def in_out_shapes(self):
        return self.in_out_shapes
    
    def sample(self, n_sample, X, seed=13):
        n = X.shape[0]
        noise = self.sample_noise(n_sample, n, seed)
        Z = self.forward(noise, X)
        return Z


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

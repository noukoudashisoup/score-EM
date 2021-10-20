"""Module containing loss functions"""
import torch
from abc import abstractmethod, ABCMeta
from scem.stein import ksd_incomplete_ustat, ksd_ustat, kcsd_ustat
from scem import util


class _Loss(metaclass=ABCMeta):
    @abstractmethod
    def loss(self, *args, **kwargs):
        pass


class KSD(_Loss):
    """

    Attributes:
        k: 
            an object either of KSTKernel,
            DKSTKernel, DKSTOnehotKernel
        score_fn:
            callable object representing
    """
    def __init__(self, k, score_fn):
        self.k = k
        self.score_fn = score_fn
    
    def loss(self, X):
        score_fn = self.score_fn
        k = self.k
        return ksd_ustat(X, score_fn, k)


class ScaledKSD(KSD):
    """KSD Scaled by a gradient penalty

    """
    def __init__(self, k, score_fn, reg=None, gradient_scale=1.):
        super(ScaledKSD, self).__init__(k, score_fn)
        self.reg = reg if reg is not None else 1e-4
        self.gradient_scale = gradient_scale
    
    def loss(self, X):
        reg = self.reg
        gsc = self.gradient_scale
        k = self.k
        scale = util.rkhs_reg_scale(X, k, reg, gsc)
        ksdsq = super(ScaledKSD, self).loss(X)
        return scale**2 * ksdsq


class IncompleteKSD(_Loss):
    """

    Attributes:
        k: 
            an object either of KSTKernel,
            DKSTKernel, DKSTOnehotKernel
        score_fn:
            callable object representing
    """
    def __init__(self, k, score_fn):
        self.k = k
        self.score_fn = score_fn
    
    def loss(self, X1, X2):
        score_fn = self.score_fn
        k = self.k
        return ksd_incomplete_ustat(X1, X2, score_fn, k)


class KCSD(_Loss):
    """KCSD U-statistics

    Attributes:
        kx:
            kernel on X
        kz:
            kernel on Z
        cond_score_fn:
            conditional score of z given x 
    """
    
    def __init__(self, kx, kz, cond_score_fn):
        self.kx = kx
        self.kz = kz
        self.cond_score_fn = cond_score_fn
    
    def loss(self, X, Z):
        kx = self.kx
        kz = self.kz
        cond_score_fn = self.cond_score_fn
        return kcsd_ustat(X, Z, cond_score_fn, kx, kz)


class VNCE:
    """VNCE of Rhodes and Gutmann, 2019.

    This class computes the VNCE loss 
    with its method `loss`. 
    This assumes that the approximte posterior
    is given by a gen.CSampelr object. 
    
    Attributes:
        csampler (gen.CSampler):
        nsampler (torch.distributions.Distribution):
        lebm (ebm.LatentEBM):
        nu (float): 
    """

    def __init__(self, csampler, nsampler, lebm, nu=1.):
        self.csampler = csampler
        self.nsampler = nsampler
        self.lebm = lebm
        self.nu = nu

    def _log_plus_one(self, x):
        return torch.log(1. + x)

    def loss(self, X, csize=10, seed=13):
        # TODO 
        # Want to reduce the forward computetion
        # to compute q(z|x) and its samples

        cs = self.csampler  # q(z|x)
        ns = self.nsampler  # py(x)
        lebm = self.lebm
        nu = self.nu

        n_data = X.shape[0]
        n_noise = int(n_data*nu)

        # avoid noise being the same
        # TODO other smart way?
        # n_noise x x_dim
        ys = ns.sample((n_noise,))

        with util.TorchSeedContext(seed):
            # csize x n x z_dim
            zs_x = cs.sample(csize, X)
            # csize x n_noise x z_dim
            zs_y = cs.sample(csize, ys)

        X_ = torch.stack([X]*csize, 0)
        Ex = lebm(X_, zs_x)
        ys_ = torch.stack([ys]*csize, 0)
        Ey = lebm(ys_, zs_y)

        log_ = self._log_plus_one

        T1 = (ns.log_prob(X) + cs.log_prob(X, zs_x) - Ex)
        T1 = -log_(nu*T1.exp()).mean(0)

        r = ((Ey - cs.log_prob(ys, zs_y)).exp().mean(0) /
             (nu * ns.log_prob(ys).exp()))
        T2 = -nu * (log_(r))
        vnce_loss = (T1 + T2).mean()

        return vnce_loss


class ConditionalKL(_Loss):
    """averaged KL divergence for training
    approximate posterior: E_x KL[q(z|x)|| p(z|x)]
    """
    def __init__(self, csampler, lebm, n_sample=1):
        self.csampler = csampler
        if not hasattr(csampler, 'log_prob'):
            raise ValueError(('{}: KL requires log density.'
                              ).format(csampler.__class__))
        self.lebm = lebm
    
    def loss(self, X, Z):
        """
        compute the loss
        """
        cs = self.csampler
        lebm = self.lebm
        log_q = cs.log_prob(X, Z)
        log_p = lebm(X, Z)
        return (log_q - log_p).mean()


class DSM(_Loss):
    pass


class MMD(_Loss):

    def __init__(self, k):
        self.k = k

    def _mmdsq_ustat(self, X, Y):
        n = X.shape[0]
        m = Y.shape[0]
        k = self.k
        Kxx = k.eval(X, X)
        Kxy = k.eval(X, Y)
        Kyy = k.eval(Y, Y)
        Kx_diag = torch.sum(torch.diag(Kxx))
        Ky_diag = torch.sum(torch.diag(Kyy))
        mmdsq = (torch.sum(Kxx) - Kx_diag)/(n*(n-1))
        mmdsq = mmdsq + (torch.sum(Kyy) - Ky_diag)/(m*(m-1))
        mmdsq = mmdsq + Kxy.mean()
        return mmdsq
    
    def loss(self, X, Y):
        return self._mmdsq_ustat(X, Y)


class ScaledMMD(MMD):

    def __init__(self, k, reg=None, gradient_scale=1.):
        super(ScaledMMD, self).__init__(k)
        self.reg = reg if reg is not None else 1e-4
        self.gradient_scale = gradient_scale

    def loss(self, X, Y):
        reg = self.reg
        gsc = self.gradient_scale
        k = self.k
        scale = util.rkhs_reg_scale(X, k, reg, gsc)
        mmdsq = super(ScaledMMD, self).loss(X, Y)
        return scale**2 * mmdsq



 

def main():
    pass

if __name__ == '__main__':
    main()

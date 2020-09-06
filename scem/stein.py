"""Module for stein discrepancies. 

This module contains functions and classes 
for computing stein discrepancies. 

"""
import torch
from scem import util
from abc import abstractmethod, ABCMeta


def ksd_ustat_gram(X, S, k):
    """Returns the gram matrix of 
    a score-absed Stein kernel

    Args:
        X (torch.Tensor): n x dx tensor
        S (torch.Tensor): n x dx tensor 
        k (kernel): a KSTKernel object

    Returns:
        torch.tensor: n x n tensor
    """
    n = X.shape[0]
    dx = X.shape[1]
    # n x dy matrix of gradients
    # n x n
    gram_score = S @ S.T
    # n x n
    K = k.eval(X, X)

    B = torch.zeros((n, n))
    C = torch.zeros((n, n))
    for i in range(dx):
        S_i = S[:, i]
        B += k.gradX_Y(X, X, i)*S_i
        C += (k.gradY_X(X, X, i).T * S_i).T

    h = K*gram_score + B + C + k.gradXY_sum(X, X)
    return h
    

def ksd_ustat(X, score_fn, k):
    n = X.shape[0]

    S = score_fn(X)
    H = ksd_ustat_gram(X, S, k)
    stat = (torch.sum(H) - torch.sum(torch.diag(H)))
    stat /= (n*(n-1))
    return stat


def kcsd_ustat(X, Z, cond_score_fn, k, l):
    n = X.shape[0]
    assert n == Z.shape[0]

    S = cond_score_fn(X, Z)
    cond_ksd_gram = ksd_ustat_gram(Z, S, l)
    K = k.eval(X, X) 
    H = K * cond_ksd_gram
    stat = (torch.sum(H) - torch.sum(torch.diag(H)))
    stat /= (n*(n-1))
    return stat


class ApproximateScore:
    """Approximate score of a latent EBM. 
    
    Given a the derivative of a joint 
    densit w.r.t. covariates, this computes
    an approximate marginal score with an 
    approximate posterior distribution over 
    the latent. 
    
    
    Attributes:
        joint_score_fn:
            Callable returning the derivative
            of p(x, z) w.r.t. x
        csampler: 
            A gen.ConditionalSampler object
            representing an approximate posterior
    """

    def __init__(self, joint_score_fn, csampler):
        self.joint_score_fn = joint_score_fn
        self.csampler = csampler

    def __call__(self, X, n_sample=100, seed=7):
        n, _ = X.shape
        cs = self.csampler
        js = self.joint_score_fn
        Z_batch = cs.sample(n_sample, X, seed=seed)
        # Assuming the last two dims are [n, dx]
        # TODO this is not efficient
        JS = [js(X, Z_batch[i]) for i in range(n_sample)]
        return torch.mean(torch.stack(JS), dim=0)

        # TODO this could be memory-intense
        # X_ = torch.stack([X.clone() for i in range(n_sample)])
        # JS = js(X_, Z_batch)
        # return torch.mean(JS, dim=0)


def main():
    import kernel
    from scem.ebm import PPCA
    seed = 13
    torch.manual_seed(seed)
    n = 100
    dx = 4
    dz = 2
    W = torch.randn([dx, dz])
    var = torch.tensor([2.0])
    ppca = PPCA(W, var)

    score_fn = ppca.score_marginal_obs
    cond_score_fn = ppca.score_joint_latent
    width_x = torch.tensor([1.0])
    width_y = torch.tensor([1.0])
    k = kernel.PTKGauss(width_x)
    l = kernel.PTKGauss(width_y)

    X = torch.randn([n, dx]) @ (W@W.T + var * torch.eye(dx))
    print(ksd_ustat(X, score_fn, k))

    Z = torch.randn([n, dz])
    X = Z @ W.T + var**0.5 * torch.randn([n, dx])
    print(kcsd_ustat(X, Z, cond_score_fn, k, l))


if __name__ == '__main__':
    main()

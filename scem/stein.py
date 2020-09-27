"""Module for stein discrepancies. 

This module contains functions and classes 
for computing stein discrepancies. 

"""
import torch
from scem import util
from abc import abstractmethod, ABCMeta


def ksd_ustat_gram(X, S, k):
    """Returns the gram matrix of 
    a score-based Stein kernel

    Args:
        X (torch.Tensor): n x dx tensor
        S (torch.Tensor): n x dx tensor 
        k (kernel): a KSTKernel/DKSTKernel object

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

    B = torch.zeros((n, n), dtype=X.dtype,
                    device=X.device)
    C = torch.zeros((n, n), dtype=X.dtype,
                    device=X.device)
    for i in range(dx):
        S_i = S[:, i]
        B += k.gradX_Y(X, X, i)*S_i
        C += (k.gradY_X(X, X, i).T * S_i).T

    h = K*gram_score + B + C + k.gradXY_sum(X, X)
    return h


def ksd_ustat(X, score_fn, k):
    """Computes KSD U-stat estimate"""
    n = X.shape[0]
    S = score_fn(X)
    H = ksd_ustat_gram(X, S, k)
    stat = (torch.sum(H) - torch.sum(torch.diag(H)))
    stat /= (n*(n-1))
    return stat


def kcsd_ustat(X, Z, cond_score_fn, k, l):
    """Computes KSCD U-stat estimate"""
    n = X.shape[0]
    assert n == Z.shape[0]
    S = cond_score_fn(X, Z)
    cond_ksd_gram = ksd_ustat_gram(Z, S, l)
    K = k.eval(X, X) 
    H = K * cond_ksd_gram
    stat = (torch.sum(H) - torch.sum(torch.diag(H)))
    stat /= (n*(n-1))
    return stat


def fssd_feat_tensor(X, V, score_fn, k):
    J, d = V.shape
    n = X.shape[0]
    assert d == X.shape[1]

    K = k.eval(X, V)  # n x J
    S = score_fn(X)
    # dKdV = torch.stack([k.gradX_Y(X, V, i) for i in range(d)])
    dKdV = torch.stack([k.gradX_y(X, V[i]) for i in range(J)])
    # dKdV = util.gradient(k.eval, 0, [X, V])
    dKdV = dKdV.permute(1, 2, 0)
    # n x d x J tensor
    SK = torch.einsum('ij,ik->ijk', S, K)
    Xi = SK + dKdV
    return Xi


def fssd_ustat(X, V, score_fn, k, return_variance=False):
    J, d = V.shape
    n = X.shape[0]
    assert d == X.shape[1]

    Xi = fssd_feat_tensor(X, V, score_fn, k)
    Tau = Xi.reshape([n, d*J])
    t1 = torch.sum(torch.mean(Tau, 0)**2)*(n/(n-1))
    t2 = (torch.sum(torch.mean(Tau**2, 0)) / n-1)
    stat = t1 - t2

    if not return_variance:
        return stat    

    mu = torch.mean(Tau, 0)
    variance = 4.*torch.mean((Tau@mu)**2) - 4*torch.sum(mu**2)**2
    return stat, variance


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

    def __init__(self, joint_score_fn, csampler, n_sample=100):
        super(ApproximateScore, self).__init__()
        self.joint_score_fn = joint_score_fn
        self.csampler = csampler
        self.n_sample = 100

    def __call__(self, X, n_sample=None, seed=7):
        n, _ = X.shape
        js = self.joint_score_fn
        ns = self.n_sample if n_sample is None else n_sample
        with torch.no_grad():
            Z_batch = self.csampler.sample(ns, X, seed=seed)
        # Assuming the last two dims are [n, dx]
        # TODO this is not efficient
        JS = [js(X, Z_batch[i]) for i in range(ns)]
        return torch.mean(torch.stack(JS), dim=0)

        # TODO this could be memory-intense
        # X_ = torch.stack([X.clone() for i in range(ns)])
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

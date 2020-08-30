"""Module for stein discrepancies. 

This module contains functions and classes 
for computing stein discrepancies. 

"""
import torch

def pscore_continuous(X, Z, energy_fn):
    """The derivate of the energy function
    (i.e., the score) is computed.

    Args:
        X (torch.Tensor):
            torch tensor of size [n, dx] 
        Z (torch.Tensor):
            torch tensor of size [n, dz]
        
        
        energy_fn (Callable[Tensor, Tensor]): energy function

    Returns:
        [torch.Tensor]:
            The derivative of energy_fn w.r.t. Z
            evaluated at X, Z. [n, dz]
        
    """
    assert isinstance(X, torch.Tensor)
    assert isinstance(Z, torch.Tensor)
    Z.requires_grad = True 
    energy = energy_fn(X, Z)
    energy_sum = torch.sum(energy)
    Gs = torch.autograd.grad(energy_sum, Z,
                             retain_graph=True,
                             only_inputs=True
                             )
    G = Gs[0]

    n, dz = Z.shape
    assert G.shape[0] == n
    assert G.shape[1] == dz
    return G

def pscore_lattice(X, Z, energy_fn):
    # TODO define
    pass


# Dictionary of posterior score functions
pscore_dict = {
    'continuous': pscore_continuous,
    'lattice': pscore_lattice,
}

def ksd_ustat_gram(X, S, k):
    n, dx = X.shape
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

def kscd_ustat(X, Z, cond_score_fn, k, l):
    n = X.shape[0]
    assert n == Z.shape[0]

    S = cond_score_fn(X, Z)
    cond_ksd_gram = ksd_ustat_gram(Z, S, l)
    K = k.eval(X, X) 
    H = K * cond_ksd_gram
    stat = (torch.sum(H) - torch.sum(torch.diag(H)))
    stat /= (n*(n-1))
    return stat

def main():
    import kernel
    from gen import PPCA
    seed = 13
    torch.manual_seed(seed)
    n = 100
    dx = 4
    dz = 2
    W = torch.randn([dx, dz])
    var = torch.tensor([2.0])
    ppca = PPCA(W, var)

    score_fn = ppca.score_marginal_obs
    cond_score_fn = ppca.posterior_score
    width_x = torch.tensor([1.0])
    width_y = torch.tensor([1.0])
    k = kernel.PTKGauss(width_x)
    l = kernel.PTKGauss(width_y)

    X = torch.randn([n, dx]) @ (W@W.T + var * torch.eye(dx))
    print(ksd_ustat(X, score_fn, k))

    Z = torch.randn([n, dz])
    X = Z @ W.T + var**0.5 * torch.randn([n, dx])
    print(kscd_ustat(X, Z, cond_score_fn, k, l))

if __name__ == '__main__':
    main()
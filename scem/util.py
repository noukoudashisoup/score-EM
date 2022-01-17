import torch
import numpy as np
from scem import kernel


class NumpySeedContext(object):
    """
    A context manager to reset the random seed by numpy.random.seed(..).
    Set the seed back at the end of the block. 
    """
    def __init__(self, seed):
        self.seed = seed 

    def __enter__(self):
        rstate = np.random.get_state()
        self.cur_state = rstate
        np.random.seed(self.seed)
        return self

    def __exit__(self, *args):
        np.random.set_state(self.cur_state)

# end NumpySeedContext


class TorchSeedContext(object):
    """
    A context manager to reset the random seed used by torch.randXXX(...)
    Set the seed back at the end of the block. 
    """
    def __init__(self, seed):
        self.seed = seed 

    def __enter__(self):
        rstate = torch.get_rng_state()
        self.cur_state = rstate
        torch.manual_seed(self.seed)
        return self

    def __exit__(self, *args):
        torch.set_rng_state(self.cur_state)

# end TorchSeedContext


def pt_dist2_matrix(X, Y=None):
    """
    Construct a pairwise Euclidean distance **squared** matrix of size
    X.shape[0] x Y.shape[0]

    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (X**2).sum(1).view(-1, 1)
    if Y is not None:
        y_norm = (Y**2).sum(1).view(1, -1)
    else:
        Y = X
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    # Some entries can be very small negative
    dist[dist <= 0] = 0.0
    return dist


def pt_meddistance(X, subsample=None, seed=283):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

    Parameters
    ----------
    X : n x d torch tensor

    Return
    ------
    median distance (a scalar, not a torch tensor)
    """
    n = X.shape[0]
    if subsample is None:
        D = torch.sqrt(pt_dist2_matrix(X, X))
        I = torch.tril_indices(n, n, -1)
        Tri = D[I[0], I[1]]
        med = torch.median(Tri)
        return med.item()
    else:
        assert subsample > 0
        with NumpySeedContext(seed=seed):
            ind = np.random.choice(n, min(subsample, n), replace=False)
        # recursion just once
        return pt_meddistance(X[ind], None, seed=seed)


def gradient(func, idx, tensors):
    """Take the gradient of func 
    w.r.t. tensors[idx].
    func is 

    Args:
        func (Callable): 
            assumed to be the function of *tensors.
        idx (int):
            index of the tensor w.r.t which
            the gradient is taken
        tensors (list or tuple ): 
            sequence of tensors of sizes [n, di]
            where di is the dims of ith tensor

    Returns:
        [type]: [description]
    """
    # variable to take grad
    X = tensors[idx]
    if X.is_leaf:
        X.requires_grad = True
    func_values = func(*tensors)
    func_sum = torch.sum(func_values)
    Gs = torch.autograd.grad(func_sum, X,
                             retain_graph=True,
                             only_inputs=True,
                             create_graph=True,
                             )
    G = Gs[0]

    # n, dx = X.shape
    # assert G.shape[0] == n
    # assert G.shape[1] == dx
    assert G.shape == X.shape
    return G


def forward_diff(func, idx,
                 tensors, lattice_ranges,
                 shift=1):
    # variable to take forward-difference
    func_values = func(*tensors)
    X = tensors[idx]
    D = torch.empty(X.shape,
                    dtype=func_values.dtype)
    d = X.shape[1]
    for j in range(d):
        X_ = tensors[idx].clone()
        X_[:, j] = (X[:, j]+shift) % lattice_ranges[j]
        tensors_ = tensors.clone()
        tensors_[idx] = X_
        D[:, j] = func(*tensors_) - func_values
    return D


def forward_diff_onehot(func, idx,
                        tensors, shift=1):
    """Takes the forward difference of 
    a specified tensor in a sequence of tensors

    Args:
        func (Callable): function of tensors
        idx (int): the index of tensor 
        tensors (seq): list or tuple of tensors
        shift (int, optional): stepsize for the difference operator.    Defaults to 1.

    Returns:
        [torch.Tensor]:
        n x d tensor 
    """
    # variable to take forward-difference
    func_values = func(*tensors)
    X = tensors[idx]
    D = torch.empty(X.shape[:-1],
                    dtype=func_values.dtype)
    _, d, K = X.shape
    perm = cyclic_perm_matrix(K, shift)

    for j in range(d):
        X_ = tensors[idx].clone()
        X_[:, j] = X[:, j] @ perm
        tensors_ = tensors.copy()
        tensors_[idx] = X_
        D[:, j] = func(*tensors_) - func_values
    return D


def cyclic_perm_matrix(n_cat, shift):
    """Obtain a cy

    Args:
        n_cat ([type]): [description]
        shift ([type]): [description]

    Returns:
        [type]: [description]
    """
    perm = torch.eye(n_cat)
    perm = perm[(torch.arange(n_cat)+shift) % n_cat]
    return perm


def rkhs_reg_scale(X, k, reg=1e-4, sc=1.):
    if not isinstance(k, kernel.KSTFuncCompose):
        raise ValueError('Currently this scaling '
                         'only supports KSTFuncCompose.')
    bk = k.k
    der = kernel.kernel_derivatives[bk.__class__]
    
    n = X.shape[0]
    gK = 0.
    f = k.f
    for j in range(f.output_shape()[0]):
        grad = f.component_grad(X, j).reshape(n, -1)
        gK += der(bk) * torch.mean(torch.sum(grad**2, axis=1))
    norm_estimate = (reg + 1. + sc*2.*gK)**0.5

    # idx = torch.arange(n)
    # K = k.pair_eval(X, X)
    # gK = k.gradXY_sum(X, X)
    # norm_estimate = (reg + K.mean() + gK[idx, idx].mean())**0.5
    return 1./norm_estimate

def sample_incomplete_ustat_batch(n, batch_size):
    cnt = 0
    idx1 = []
    idx2 = []
    while cnt < batch_size:
        i1 = torch.randint(0, n, [batch_size])
        i2 = torch.randint(0, n, [batch_size])
        comp_idx = (i1 < i2)
        cnt += torch.sum(comp_idx)
        idx1.append(i1[comp_idx])
        idx2.append(i2[comp_idx])
    idx1 = torch.cat(idx1)[:batch_size]
    idx2 = torch.cat(idx2)[:batch_size]
    return idx1, idx2


def pnorm(X, p, dim=None, keepdim=False, axis=None):
    if p < 0:
        raise ValueError('Power has to be positive. Was {}'.format(p))
    if axis is not None:
        dim = axis
    return torch.sum(X**p, dim=dim, keepdim=keepdim) ** (1./p)

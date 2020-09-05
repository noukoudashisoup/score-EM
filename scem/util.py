import torch
import numpy as np


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
                             only_inputs=True
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
        tensors_ = tensors.copy()
        tensors_[idx] = X_
        D[:, j] = func(*tensors_) - func_values
    return D

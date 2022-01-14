import torch
import copy
from scipy.special import comb


def poly_unisolvent_points(d, deg):
    """Returns a dictionary of a unisolvent set for 
    the set of polynomials of degree deg

    Args:
        d (int): dimension
        deg (int): the maximum degree of monomials

    Raises:
        ValueError: if deg > 2

    Returns:
        dict: a dictionary of p tensors with p = (d+deg) chooses d.
            Each point is a tensor of size (1, d)
    """
    if deg > 2:
        raise ValueError("Only deg <= 2 is supported.")

    unisolvent_sets = {}
    if deg >= 0:
        unisolvent_sets[0] = torch.zeros((1, d))
    if deg >= 1:
        eye_d = torch.eye(d)
        for i in range(d):
            unisolvent_sets[i+1] = eye_d[None, i]
    if deg == 2:
        indices = torch.triu_indices(d, d)
        eye_d = torch.eye(d)
        for i in range(len(indices[0])):
            i1, i2 = int(indices[0, i]), int(indices[1, i])
            if i1 == i2:
                unisolvent_sets[(i1, i1)] = 2. * eye_d[None, i1]
            else:
                x = torch.zeros((1, d,))
                x[:, i1] = x[:, i2] = 1.
                unisolvent_sets[(i1, i2)] = x
    return unisolvent_sets


def lagrange_basis(X, deg):
    """Computes the lagrange basis corresponding to 
    the point set X_m^d = {d-multi-indices with 
    at most m elements being one and the others 0}.
    The basis includes the constant term (which is just one).

    Args:
        X (torch.Tensor): n x d tensor
        deg (int): the maximum degree of monomials

    Raises:
        ValueError: if order > 2

    Returns:
        torch.Tensor: n x p tensor with p = (d + deg) chooses d
    """
    if deg > 2:
        raise ValueError('This method does not support higher order polynomials. '
                         'The value was {}'.format(deg))
    n, d = X.shape
    p = int(comb(d+deg, d))
    basis_funcs_on_X = torch.empty((n, p),
                                   dtype=X.dtype, device=X.device)
    basis_funcs_on_X[:, 0] = 1
    if deg == 1:
        basis_funcs_on_X[:, 1: d+1] = X
        return basis_funcs_on_X
    if deg == 2:
        second_order_terms = torch.einsum('ij, ik -> ijk', X, X)
        for j in range(d):
            X_j = X[:, j]
            X_j_tile = torch.tile(X[:, j, None], (1, d-1))
            idx = torch.arange(d)
            idx = idx[idx!=j]
            A = X_j * (X_j_tile - X[:, idx]).prod(axis=1)
            B = X_j * (1. - X[:, idx]).prod(axis=1)
            basis_funcs_on_X[:, 1+j] = (A - 2**(d-1) * B) / (1.-2**(d-1))
            second_order_terms[:, j, j] = (A - B) / (2**d - 2)
        triu_idx = torch.triu_indices(d, d)
        basis_funcs_on_X[:, d+1:] = (second_order_terms[:, triu_idx[0], triu_idx[1]]
                                     ).reshape(n, -1)
        basis_funcs_on_X[:, 0] -= basis_funcs_on_X[:, 1:].sum(axis=1)
        return basis_funcs_on_X


def _second_order_lagrange_basis_dict(dim, deg):

    def inter_mediate(X, j):
        X_j = X[:, j]
        X_j_tile = torch.tile(X[:, j, None], (1, dim-1))
        idx = torch.arange(dim)
        idx = idx[idx!=j]
        A = X_j * (X_j_tile - X[:, idx]).prod(axis=1)
        B = X_j * (1. - X[:, idx]).prod(axis=1)
        return A, B
    
    def first_order_func_i(i):
        def func_i(X):
            A, B = inter_mediate(X, i)
            return (A - 2**(dim-1) * B) / (1.-2**(dim-1))
        return func_i

    def second_order_func_ii(i):
        def func_ii(X):
            A, B = inter_mediate(X, i)
            return (A - B) / (2**dim - 2)
        return func_ii

    def second_order_func_ij(i, j):
        def func_ij(X):
            return X[:, i] * X[:, j]
        return func_ij
    
    def one_minus_funcs(func_dict):
        def minused(X):
            n = X.shape[0]
            evals = torch.ones((n,), dtype=X.dtype, device=X.device)
            for fn in func_dict.values():
                evals -= fn(X)
            return evals
        return minused

    basis_dict = {}
    for i in range(dim):
        basis_dict[i+1] = first_order_func_i(i)

    triu_idx = torch.triu_indices(dim, dim)
    len_second_order_terms = len(triu_idx[0])
    for i in range(len_second_order_terms):
        i1, i2 = int(triu_idx[0, i]), int(triu_idx[1, i])
        if i1 == i2:
            basis_dict[(i1, i1)] = second_order_func_ii(i1)
        else:
            basis_dict[(i1, i2)] = second_order_func_ij(i1, i2)
    basis_dict[0] = one_minus_funcs(copy.deepcopy(basis_dict))
    return basis_dict


def build_lagrange_basis_dict(dim, deg):
    if deg > 2:
        raise ValueError('This method does not support higher order polynomials. '
                         'The value was {}'.format(deg))
    basis_dict = {}
    if deg == 0:
        basis_dict[0] = (lambda X: 
            torch.ones((X.shape[0],), device=X.device, dtype=X.dtype))
    if deg == 1:
        def return_ith_col(i):
           return lambda X: X[:, i]
        basis_dict[0] = lambda X: 1. - X.sum(axis=1)
        for i in range(dim):
            basis_dict[i+1] = return_ith_col(i)
    if deg == 2:
        basis_dict = _second_order_lagrange_basis_dict(dim, deg)
    return basis_dict  

def main():
    d = 5
    deg = 2
    unisolv = poly_unisolvent_points(d, deg)
    basis_dict = build_lagrange_basis_dict(d, deg)
    arange_d = torch.arange(d)
    for key, point in unisolv.items():
        for fn_key, fn in basis_dict.items():
            evals = fn(point).item()
            if (evals != 0):
                print(key == fn_key)
    # evals = (lagrange_basis(X, deg=2))
    # n = int(comb(d+deg, d))
    # arange_n = torch.arange(n)
    # for i in range(n):
    #     idx = (evals[i]!=0)
    #     print(arange_n[idx], evals[i][idx])
    # print(lagrange_basis(torch.randn(1, d), deg=2))

if __name__ == '__main__':
    main()
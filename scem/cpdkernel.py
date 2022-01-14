"""
A module containing conditionally positive definite kernels.
This module is separated from 'kernel' module to
make the non-positive definiteness explicit.
"""

import torch
from math import ceil
from scem import util
from abc import ABCMeta
from abc import abstractmethod


class KSTKernelCPD(metaclass=ABCMeta):
    """
    Interface specifiying methods a kernel has to implement to be used with
    the Kernelized Stein discrepancy.
    We require some additional method required for composing a deep
    kernel.
    """
    @abstractmethod
    def order(self):
        pass

    @abstractmethod
    def gradX(self, X, Y):
        pass

    def gradX_pair(self, X, Y):
        assert X.shape == Y.shape
        return util.gradient(self.pair_eval, 0, [X, Y])

    def gradY(self, X, Y):
        dims = list(range(len(X.shape)))
        dims[0] = 1
        dims[1] = 0
        return self.gradX(Y, X).transpose(*dims)

    @abstractmethod
    def gradXY_sum(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\par^2 k(x, Y)}{\par x_i \par y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
        """
        pass

    @abstractmethod
    def gradXY_sum_pair(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\par^2 k(x, y)}{\par x \par y}
        evaluated at each x_i in X, and y_i in Y.

        X: n x d Tensor.
        Y: n x d Tensor.

        Return a [n, ] Tensor of the derivatives.
        """
        pass


class MultiQuadratic(KSTKernelCPD):

    """Class of Multi-Quadratic (MQ) kernels.
    """

    def __init__(self, b=0.5, c=1.0, s2=1.0):
        if (b <= 0) or b.is_integer():
            raise ValueError(
                "b has to be positive and not an integer. Was {}".format(b))
        if not c > 0:
            raise ValueError("c has to be positive. Was {}".format(c))
        self.b = b
        self.c = c
        self.s2 = (torch.tensor(s2) if not isinstance(s2, torch.Tensor)
                   else s2)
    
    def order(self):
        return ceil(self.b)

    def eval(self, X, Y):
        b = self.b
        c = self.c
        s2 = torch.sqrt(self.s2)**2
        D2 = util.pt_dist2_matrix(X, Y)
        K = (-1)**ceil(b) * (c ** 2 + D2/s2) ** b
        return K

    def pair_eval(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        b = self.b
        c = self.c
        s = torch.sqrt(self.s2)
        return (-1)**ceil(b) * (c ** 2 + torch.sum(((X - Y)/s) ** 2, 1)) ** b

    def gradX(self, X, Y):
        d = X.shape[1]
        G = torch.stack([self.parX(X, Y, j) for j in range(d)])
        return G.permute(1, 2, 0)

    def gradX_pair(self, X, Y):
        b = self.b
        c = self.c
        diff = X - Y
        s2 = torch.sqrt(self.s2)**2
        D2 = torch.sum((diff)**2, axis=1)
        G = (-1)**ceil(b) * torch.einsum('ij,i->ij', diff/s2, 2.0*b*(c**2 + D2/s2)**(b-1))
        return G
    
    def parX(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        s2 = torch.sqrt(self.s2)**2
        D2 = util.pt_dist2_matrix(X, Y)
        # 1d array of length nx
        Xi = X[:, dim]
        # 1d array of length ny
        Yi = Y[:, dim]
        # nx x ny
        dim_diff = (Xi.unsqueeze(1) - Yi.unsqueeze(0))
        assert dim_diff.shape == (X.shape[0], Y.shape[0])

        b = self.b
        c = self.c
        Gdim = (-1)**ceil(b) * ( 2.0*b*(c**2 + D2/s2)**(b-1) )*dim_diff / s2
        assert Gdim.shape[0] == X.shape[0]
        assert Gdim.shape[1] == Y.shape[0]
        return Gdim

    def gradXY_sum(self, X, Y):
        """
        Compute
        \sum_{i=1}^d \frac{\par^2 k(X, Y)}{\par x_i \par y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
        """
        s2 = torch.sqrt(self.s2)**2
        b = self.b
        c = self.c
        D2 = util.pt_dist2_matrix(X, Y)

        # d = input dimension
        d = X.shape[1]
        c2D2 = c**2 + D2/s2
        T1 = -4.0*b*(b-1)*D2*(c2D2**(b-2)) / (s2**2)
        T2 = -2.0*b*d*c2D2**(b-1) / s2
        return (-1)**ceil(b) * (T1 + T2)

    def gradXY_sum_pair(self, X, Y):
        s2 = torch.sqrt(self.s2)**2
        b = self.b
        c = self.c
        D2 = torch.sum((X-Y)**2, axis=1)

        # d = input dimension
        d = X.shape[1]
        c2D2 = c**2 + D2/s2
        T1 = -4.0*b*(b-1)*D2*(c2D2**(b-2)) / (s2**2)
        T2 = -2.0*b*d*c2D2**(b-1) / s2
        return (-1)**ceil(b) * (T1 + T2)

    def order(self):
        return ceil(self.b)


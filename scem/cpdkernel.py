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

        
        Y: n x d Tensor.

        Return a [n, ] Tensor of the derivatives.
        """
        pass


class CKSTPrecondionedMQ(KSTKernelCPD):

    """Precontioned Multi-Quadratic kernel
        k(x,y) = (c^2 + <P(x-y), P(x-y)>)^b
        Note that the input has to have a compatible dimension with P. 

        Args:
            c (float): a positive bias parameter
            b (float): the exponenent (positive)
            P (torch.tensor): square root of a preconditioning matrix. 
                Required to be positive definite.
    """
    def __init__(self, b=0.5, c=1.0, P=None):
        if (b <= 0) or b.is_integer():
            raise ValueError(
                "b has to be positive and not an integer. Was {}".format(b))
        if not c > 0:
            raise ValueError("c has to be positive. Was {}".format(c))
        if P is None:
            P = torch.eye(self.dim)
        self.b = b
        self.c = c
        self.P = P
        _, s, _ = torch.linalg.svd((P @ P))
        if torch.min(s) <= 1e-8:
            raise ValueError('P has to be positive definite')

    def order(self):
        return ceil(self.b)

    def _load_params(self):
        return self.c, self.b, self.P

    def eval(self, X, Y):
        """Evalute the kernel on data X and Y """
        c, b, P = self._load_params()
        X_ = X @ P.T
        Y_ = Y @ P.T
        D2 = util.pt_dist2_matrix(X_, Y_)
        K = (-1)**ceil(b) * (c**2 + D2)**b
        return K
    
    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...
        """
        assert X.shape[0] == Y.shape[0]
        c, b, P = self._load_params()
        X_ = X @ P.T
        Y_ = Y @ P.T

        K = (-1)**ceil(b) * (c**2 + torch.sum((X_-Y_)**2, 1))**b
        return K

    def gradX(self, X, Y):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        c, b, P = self._load_params()
        X_ = X @ P.T
        Y_ = Y @ P.T
        D2 = util.pt_dist2_matrix(X_, Y_)
        diff = (X_[:, None] - Y_[None])
        Gdim = ( 2.0*b*(c**2 + D2)**(b-1) )[:, :, None] * diff
        Gdim = (-1)**ceil(b) * Gdim @  P
        assert Gdim.shape[0] == X.shape[0]
        assert Gdim.shape[1] == Y.shape[0]
        return Gdim

    def gradX_pair(self, X, Y):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        c, b, P = self._load_params()
        X_ = X @ P.T
        Y_ = Y @ P.T
        diff = (X_ - Y_)
        D2 = torch.sum(diff**2, axis=1)
        Gdim = diff * ( 2.0*b*(c**2 + D2)**(b-1) )[:, None] 
        Gdim = (-1)**ceil(b) * (Gdim @  P)
        return Gdim

    def gradXY_sum(self, X, Y):
        """
        Compute
        \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: (torch.tensor) nx x d tensor
        Y: (torch.tensor) ny x d tensor

        Return a nx x ny tensor of the derivatives.
        """
        c, b, P = self._load_params()
        P_ = P @ P.T
        X_ = X @ P.T
        Y_ = Y @ P.T
        diff = (X_[:, None] - Y_[None])
        D2 = util.pt_dist2_matrix(X_, Y_)

        c2D2 = c**2 + D2
        T1 = torch.einsum('ij, nmi, nmj->nm', P_, diff, diff)
        T1 = -T1 * 4.0*b*(b-1)*(c2D2**(b-2))
        T2 = -2.0*b*torch.trace(P_) * c2D2**(b-1)
        return (-1)**ceil(b) * ( T1 + T2 )

    def gradXY_sum_pair(self, X, Y):
        """
        Compute
        \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: (torch.tensor) n x d tensor
        Y: (torch.tensor) n x d tensor

        Return a [n,] tensor of the derivatives.
        """
        c, b, P = self._load_params()
        P_ = P @ P.T
        X_ = X @ P.T
        Y_ = Y @ P.T
        diff = (X_ - Y_)
        D2 = torch.sum(diff**2, axis=-1)

        c2D2 = c**2 + D2
        T1 = torch.einsum('ij, ni, nj->n', P_, diff, diff)
        T1 = -T1 * 4.0*b*(b-1)*(c2D2**(b-2))
        T2 = -2.0*b*torch.trace(P_) * c2D2**(b-1)
        return (-1)**ceil(b) * (T1 + T2)


# end class CKSTPrecondionedMQ

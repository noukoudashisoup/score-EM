"""
A module containing kernel functions.
"""

import torch
from scem import util
from abc import ABCMeta
from abc import abstractmethod


class Kernel(object, metaclass=ABCMeta):
    """Abstract class for kernels. Inputs to all methods are numpy arrays."""

    @abstractmethod
    def eval(self, X, Y):
        """
        Evaluate the kernel on data X and Y
        X: nx x d where each row represents one point
        Y: ny x d
        return nx x ny Gram matrix
        """
        pass

    @abstractmethod
    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...

        X: n x d where each row represents one point
        Y: n x d
        return a 1d numpy array of length n.
        """
        pass


class KSTKernel(Kernel, metaclass=ABCMeta):
    """
    Interface specifiying methods a kernel has to implement to be used with
    the Kernelized Stein discrepancy test of Chwialkowski et al., 2016 and
    Liu et al., 2016 (ICML 2016 papers) See goftest.KernelSteinTest.
    """

    @abstractmethod
    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        pass

    def gradY_X(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of Y in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        return self.gradX_Y(Y, X, dim).T

    @abstractmethod
    def gradXY_sum(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\partial^2 k(x, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
        """
        pass

    def gradX_y(self, X, y):
        n = X.shape[0] 
        Y = torch.stack([y]*n)
        return util.gradient(self.eval, 0, [X, Y])

# end KSTKernel


class DKSTKernel(Kernel):
    """
    Interface specifiying methods a kernel has to implement to be used with 
    the Discrete Kernelized Stein discrepancy test of Yang et al., 2018.
    """

    def __init__(self, lattice_ranges, d):
        """Require subclasses to have n_values and d """
        self.lattice_ranges = lattice_ranges
        self._d = d
    
    def gradX_Y(self, X, Y, dim, shift=-1):
        """
        Default: compute the (cyclic) backward difference with respect to 
        the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """

        X_ = X.clone()
        X_[:, dim] = torch.remainder(X_[:, dim]+shift, self.lattice_ranges[dim])
        return (self.eval(X, Y) - self.eval(X_, Y))

    def gradY_X(self, X, Y, dim, shift=-1):
        """
        Default: compute the (cyclic) backward difference with respect to 
        the dimension dim of Y in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        return self.gradX_Y(Y, X, dim, shift).T

    def gradXY_sum(self, X, Y, shift=-1):
        """
        Compute the trace term in the kernel function in Yang et al., 2018. 

        X: nx x d numpy array.
        Y: ny x d numpy array. 

        Return a nx x ny numpy array of the derivatives.
        """
        nx, d = X.shape
        ny, _ = Y.shape
        K = torch.zeros((nx, ny), dtype=X.dtype)
        lattice_ranges = self.lattice_ranges
        for j in range(d):
            X_ = X.clone()
            Y_ = Y.clone()
            X_[:, j] = (X[:, j]+shift % lattice_ranges[j])
            Y_[:, j] = (Y[:, j]+shift % lattice_ranges[j])
            K += (self.eval(X, Y) + self.eval(X_, Y_)
                  - self.eval(X_, Y) - self.eval(X, Y_))
        return K

    def dim(self):
        return self._d

# end DKSTKernel


class PTKGauss(KSTKernel):
    """
    Pytorch implementation of the isotropic Gaussian kernel.
    Parameterization is the same as in the density of the standard normal
    distribution. sigma2 is analogous to the variance.
    """

    def __init__(self, sigma2):
        """
        sigma2: a number representing squared width
        """
        assert (sigma2 > 0).any(), 'sigma2 must be > 0. Was %s'%str(sigma2)
        self.sigma2 = sigma2

    def eval(self, X, Y):
        """
        Evaluate the Gaussian kernel on the two 2d Torch Tensors

        Parameters
        ----------
        X : n1 x d Torch Tensor
        Y : n2 x d Torch Tensor

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        sigma2 = torch.sqrt(self.sigma2**2)
        sumx2 = torch.sum(X**2, dim=1).view(-1, 1)
        sumy2 = torch.sum(Y**2, dim=1).view(1, -1)
        D2 = sumx2 - 2*torch.matmul(X, Y.transpose(1, 0)) + sumy2
        K = torch.exp(-D2.div(2.0*sigma2))
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d Pytorch tensors

        Return
        -------
        a Torch tensor with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        D2 = torch.sum( (X-Y)**2, 1)
        sigma2 = torch.sqrt(self.sigma2**2)
        Kvec = torch.exp(old_div(-D2, (2.0*sigma2)))
        return Kvec

    def __str__(self):
        return "PTKGauss(%.3f)" % self.sigma2

    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a Torch tensor of size nx x ny.
        """
        sigma2 = self.sigma2
        K = self.eval(X, Y)
        Diff = X[:, [dim]] - Y[:, [dim]].T
        #Diff = np.reshape(X[:, dim], (-1, 1)) - np.reshape(Y[:, dim], (1, -1))
        G = -K*Diff/sigma2
        return G

    def gradXY_sum(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny Torch tensor of the derivatives.
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        d = d1
        sigma2 = self.sigma2
        D2 = torch.sum(X**2, 1).view(n1, 1) - 2*torch.matmul(X, Y.T) + torch.sum(Y**2, 1).view(1, n2)
        K = torch.exp(-D2/(2.0*sigma2))
        G = K/sigma2 *(d - D2/sigma2)
        return G


class KIMQ(KSTKernel):
    """Class of Inverse MultiQuadratic (IMQ) kernels.
    Have not been tested. Be careful. 
    """

    def __init__(self, b=-0.5, c=1.0, s2=1.0):
        if not b < 0:
            raise ValueError("b has to be negative. Was {}".format(b))
        if not c > 0:
            raise ValueError("c has to be positive. Was {}".format(c))
        self.b = b
        self.c = c
        self.s2 = s2

    def eval(self, X, Y):
        b = self.b
        c = self.c
        s2 = self.s2
        sumx2 = torch.sum(X ** 2, 1).reshape(-1, 1)
        sumy2 = torch.sum(Y ** 2, 1).reshape(1, -1)
        # D2 = sumx2 - 2.0 * X.mm(Y.t()) + sumy2
        D2 = util.pt_dist2_matrix(X, Y)
        K = (c ** 2 + D2/s2**2) ** b
        return K

    def pair_eval(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        b = self.b
        c = self.c
        s = self.s2**0.5
        return (c ** 2 + torch.sum(((X - Y)/s) ** 2, 1)) ** b

    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        s2 = self.s2
        D2 = util.pt_dist2_matrix(X, Y)
        # 1d array of length nx
        Xi = X[:, dim]
        # 1d array of length ny
        Yi = Y[:, dim]
        # nx x ny
        dim_diff = (Xi.unsqueeze(1) - Yi.unsqueeze(0))

        b = self.b
        c = self.c
        Gdim = ( 2.0*b*(c**2 + D2)**(b-1) )*dim_diff / s2
        assert Gdim.shape[0] == X.shape[0]
        assert Gdim.shape[1] == Y.shape[0]
        return Gdim

    def gradXY_sum(self, X, Y):
        """
        Compute
        \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
        """
        s2 = self.s2
        b = self.b
        c = self.c
        D2 = util.pt_dist2_matrix(X, Y)

        # d = input dimension
        d = X.shape[1]
        c2D2 = c**2 + D2/s2
        T1 = -4.0*b*(b-1)*D2*(c2D2**(b-2)) / (s2**2)
        T2 = -2.0*b*d*c2D2**(b-1) / s2
        return T1 + T2


class KHamming(DKSTKernel):

    def __init__(self, lattice_ranges):
        """
        Args:
        - lattice_ranges: a positive integer/ integer array specifying
        the number of possible of the discrete variable. 
        """
        self.lattice_ranges = lattice_ranges
        self.d = len(lattice_ranges)

    def eval(self, X, Y):
        """
        Evaluate the kernel on data X and Y
        Args: 
            X: n x d where each row represents one point
            Y: n x d
        Return: 
            a n x n numpy array.
        """
        assert X.shape[1] == Y.shape[1]
        d = self.d
        hamm_dist = torch.cdist(X, Y, 0) / d
        return torch.exp(-hamm_dist)

    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...
        
        X: n x d where each row represents one point
        Y: n x d
        return a 1d numpy array of length n.
        """
        assert X.shape == Y.shape
        n, d = X.shape
        H = torch.zeros((n, d), dtype=X.dtype)
        H[X!=Y] = 1
        return torch.exp(-torch.mean(H, dim=1))


class DKSTOnehotKernel(Kernel):
    """
    Interface specifiying methods a kernel has to implement to be used with 
    the Discrete Kernelized Stein discrepancy test of Yang et al., 2018.

    The input is expected to be categorical variables represented
    by onehot encoding.
    """

    def __init__(self, n_cat):
        self.n_cat = n_cat
    
    def gradX_Y(self, X, Y, dim, shift=-1):
        """
        Default: compute the (cyclic) backward difference with respect to
        the dimension dim of X in k(X, Y).

        X: nx x d x n_cat
        Y: ny x d x n_cat

        Return a numpy array of size nx x ny.
        """

        X_ = X.clone()
        perm = util.cyclic_perm_matrix(self.n_cat, shift)
        X_[:, dim] = X[:, dim] @ perm
        return (self.eval(X, Y) - self.eval(X_, Y))

    def gradY_X(self, X, Y, dim, shift=-1):
        """
        Default: compute the (cyclic) backward difference with respect to 
        the dimension dim of Y in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        return self.gradX_Y(Y, X, dim, shift).T

    def gradXY_sum(self, X, Y, shift=-1):
        """
        Compute the trace term in the kernel function in Yang et al., 2018. 

        X: nx x d x n_cat numpy array.
        Y: ny x d x n_cat numpy array. 

        Return a nx x ny numpy array of the derivatives.
        """
        nx = X.shape[0]
        d = X.shape[1]
        ny = Y.shape[0]
        K = torch.zeros((nx, ny), dtype=X.dtype)
        perm = util.cyclic_perm_matrix(self.n_cat, shift)
        for j in range(d):
            X_ = X.clone()
            Y_ = Y.clone()
            X_[:, j] = X[:, j] @ perm
            Y_[:, j] = Y[:, j] @ perm
            K += (self.eval(X, Y) + self.eval(X_, Y_)
                  - self.eval(X_, Y) - self.eval(X, Y_))
        return K

# end DKSTKernel


class OHKGauss(DKSTOnehotKernel):
    """Gaussian kernel defined on
    categorical variables in onehot representation. 

    Attributes:
        n_cat (int): 
            The number of categories of the input
            variable
        sigma2 (torch.tensor):
            The bandwidth paramter. A scalar or vector.
    """

    def __init__(self, n_cat, sigma2):
        super(OHKGauss, self).__init__(n_cat)
        self.sigma2 = sigma2

    def eval(self, X, Y):
        """
        Evaluate the kernel on data X and Y
        Args: 
            X: n x d x n_cat where each row represents one point
            Y: n x d x n_cat
        Return: 
            a n x n numpy array.
        """
        assert X.shape[1] == Y.shape[1]
        assert X.shape[2] == X.shape[2]
        nx = X.shape[0]
        ny = Y.shape[0]
        X_ = X.reshape(nx, -1)
        Y_ = Y.reshape(ny, -1)
        D2 = util.pt_dist2_matrix(X_, Y_)
        return torch.exp(-D2/self.sigma2)

    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...
        
        X: n x d x n_cat where each row represents one point
        Y: n x d x n_cat
        return a 1d numpy array of length n.
        """
        assert X.shape == Y.shape
        n = X.shape[0]
        diff2 = (X - Y)**2
        d2sum = torch.exp(-torch.sum(diff2.view(n, -1), axis=1))
        return torch.exp(-d2sum/self.sigma2)

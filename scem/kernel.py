"""
A module containing kernel functions.
"""

import torch
from scem import util
from math import ceil
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


class DifferentiableKernel(Kernel):
    """Class representing differentiable kernels.
    All subclasses should have a prefix B (B stands for Base kernels).
    """

    def gradX_pair(self, X, Y):
        assert X.shape == Y.shape
        return util.gradient(self.pair_eval, 0, [X, Y])
    
    def gradY_pair(self, X, Y):
        assert X.shape == Y.shape
        return util.gradient(self.pair_eval, 1, [X, Y])

    def gradX_y(self, X, y):
        n = X.shape[0] 
        Y = torch.stack([y]*n)
        return util.gradient(self.pair_eval, 0, [X, Y])
    
    @abstractmethod
    def gradX(self, X, Y):
        """Returns nabla_x k(x, y)|_{x=X[i], Y[j]}"""
        pass
        
    def gradY(self, X, Y):
        """Returns nabla_y k(x, y)|_{x=X[i], Y[j]}"""
        dims = list(range(len(X.shape)))
        dims[0] = 1
        dims[1] = 0
        return self.gradX(Y, X).permute(*dims)
        
    @abstractmethod
    def gradXY(self, X, Y):
        """Returns nabla_x nabla_y k(x, y)|_{x=X[i], Y[j]}"""
        pass

    @abstractmethod
    def gradXY_pair(self, X, Y):
        """Returns nabla_x nabla_y k(x, y)|_{x=X[i], Y[i]}"""
        pass


class KSTKernel(Kernel, metaclass=ABCMeta):
    """
    Interface specifiying methods a kernel has to implement to be used with
    the Kernelized Stein discrepancy.
    We require some additional method required for composing a deep
    kernel.
    """

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
    
    def parX(self, X, Y, dim, shift=-1):
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

    def parY(self, X, Y, dim, shift=-1):
        """
        Default: compute the (cyclic) backward difference with respect to 
        the dimension dim of Y in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        return self.parX(Y, X, dim, shift).T
    
    def gradX(self, X, Y, shift=-1):
        d = X.shape[1]
        G = torch.stack([self.parX(X, Y, j, shift) for j in range(d)])
        return G.permute(1, 2, 0)

    def gradY(self, X, Y, shift=-1):
        dims = list(range(len(X.shape)))
        dims[0] = 1
        dims[1] = 0
        return self.gradX(Y, X).permute(*dims)

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


class DKSTOnehotKernel(Kernel):
    """
    Interface specifiying methods a kernel has to implement to be used with 
    the Discrete Kernelized Stein discrepancy test of Yang et al., 2018.

    The input is expected to be categorical variables represented
    by onehot encoding.
    """

    def __init__(self, n_cat):
        self.n_cat = n_cat
    
    def parX(self, X, Y, dim, shift=-1):
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

    def parY(self, X, Y, dim, shift=-1):
        """
        Default: compute the (cyclic) backward difference with respect to 
        the dimension dim of Y in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        return self.parX(Y, X, dim, shift).T

    def gradX(self, X, Y, shift=-1):
        """Compute the gradient (difference) w.r.t. X."""
        d = X.shape[1]
        G = torch.stack([self.parX(X, Y, j, shift) for j in range(d)])
        return G.permute(1, 2, 0)

    def parX_pair(self, X, Y, dim, shift=-1):
        """
        Default: compute the (cyclic) backward difference with respect to
        the dimension dim of X in k(X, Y) pairwise.

        X: n x d x n_cat
        Y: n x d x n_cat

        Return a numpy array of size n.
        """

        X_ = X.clone()
        perm = util.cyclic_perm_matrix(self.n_cat, shift)
        X_[:, dim] = X[:, dim] @ perm
        return (self.pair_eval(X, Y) - self.pair_eval(X_, Y))

    def gradX_pair(self, X, Y, shift=-1):
        """Compute the gradient (difference) w.r.t. X pairwise"""
        d = X.shape[1]
        G = torch.stack([self.parX_pair(X, Y, j, shift) for j in range(d)])
        return G.permute(1, 0)

    def gradXY_sum(self, X, Y, shift=-1):
        """
        Compute the trace term in the kernel function in Yang et al., 2018. 

        X: nx x d x n_cat numpy array.
        Y: ny x d x n_cat numpy array. 

        Return a nx x ny torch tensor of the derivatives.
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
    
    def gradXY_sum_pair(self, X, Y, shift=-1):
        """
        Compute the trace term in the kernel function in Yang et al., 2018
        (pairwise).

        X: n x d x n_cat numpy array.
        Y: n x d x n_cat numpy array. 

        Return a torch tensor of the derivatives, size [n].
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
            K += (self.pair_eval(X, Y) + self.pair_eval(X_, Y_)
                  - self.pair_eval(X_, Y) - self.pair_eval(X, Y_))
        return K
    
# end DKSTOnehotKernel


class FeatureMap(metaclass=ABCMeta):
    """
    Abstract class for a feature map of a kernel.
    Assume the map is differentiable.
    """

    @abstractmethod
    def __call__(self, x):
        """
        Return a feature vector for the input x.
        """
        raise NotImplementedError()

    @abstractmethod
    def input_shape(self):
        """
        Return the expected input shape of this feature map (excluding the
        batch dimension).  For instance if each point is a 32x32 pixel image,
        then return (32, 32).
        """
        raise NotImplementedError()

    @abstractmethod
    def output_shape(self):
        """
        Return the output shape of this feature map.
        """
        raise NotImplementedError()

    def component_grad(self, X, idx):
        def f_(X):
            return self(X)[:, idx]
        return util.gradient(f_, 0, [X])

# end class FeatureMap


class FuncFeatureMap(FeatureMap):
    def __init__(self, f, in_shape, out_shape):
        """
        f: a callable object representing the feature map.
        in_shape: expected shape of the input
        out_shape: expected shape of the output
        """
        self.f = f
        self.in_shape = in_shape
        self.out_shape = out_shape

    def __call__(self, x):
        f = self.f
        return f(x)

    def input_shape(self):
        return self.in_shape

    def output_shape(self):
        return self.out_shape

 
class KFuncCompose(Kernel):
    """
    A kernel given by k'(x,y) = k(f(x), f(y)), where f is the specified 
    function, and k is the specified kernel.
    f has to be callable.
    """

    def __init__(self, k, f):
        """
        k: a PTKernel
        f: a callable object or a function
        """
        self.k = k
        self.f = f

    def eval(self, X, Y):
        f = self.f
        k = self.k
        fx = f(X)
        fy = f(Y)
        return k.eval(fx, fy)

    def pair_eval(self, X, Y):
        f = self.f
        k = self.k
        fx = f(X)
        fy = f(Y)
        return k.pair_eval(fx, fy)


class KSTFuncCompose(KFuncCompose, KSTKernel):

    def __init__(self, k, f):
        if not isinstance(k, DifferentiableKernel):
            raise TypeError('k needs to be differentiable')
        if not isinstance(f, FuncFeatureMap):
            raise TypeError('feature function needs to be differentiable')
        super(KSTFuncCompose, self).__init__(k, f)
    
    def gradX(self, X, Y):
        k = self.k
        f = self.f
        nx = X.shape[0]
        ny = Y.shape[0]
        dx = X[0].numel()

        assert len(f.output_shape()) == 1
        d_out = f.output_shape()[0]

        # n1 x n2 x d_out
        kG = k.gradX(f(X), f(Y))

        g_feat_X = torch.empty((d_out, nx, dx),
                               dtype=X.dtype,
                               device=X.device)
        for i_o in range(d_out):
            g_feat_X[i_o] = f.component_grad(X, i_o).reshape(nx, dx)
        
        G = torch.einsum('nmi,inj->nmj', kG, g_feat_X)
        return G.reshape((nx, ny,)+X[0].shape)


    def gradXY_sum(self, X, Y):
        assert isinstance(self.f, FuncFeatureMap)
        k = self.k
        f = self.f
        assert len(f.output_shape()) == 1
        d_out = f.output_shape()[0]
        nx = X.shape[0]
        ny = Y.shape[0]
        dx = X[0].numel()
        dy = Y[0].numel()
        assert dx == dy

        # n1 x n2 x d_out x d_out
        kG = k.gradXY(f(X), f(Y))

        g_feat_X = torch.empty((d_out, nx, dx),
                               dtype=X.dtype,
                               device=X.device)
        for i_o in range(d_out):
            g_feat_X[i_o] = f.component_grad(X, i_o).reshape(nx, dx)

        if Y is X:
            G = torch.einsum('ink, nmij, jmk->nm', g_feat_X, kG, g_feat_X)
            return G

        g_feat_Y = torch.empty((d_out, ny, dy),
                               dtype=Y.dtype,
                               device=Y.device)
        for i_o in range(d_out):
            g_feat_Y[i_o] = f.component_grad(Y, i_o).reshape(ny, dy)

        G = torch.einsum('ink, nmij, jmk->nm', g_feat_X, kG, g_feat_Y)
        return G

    def gradXY_sum_pair(self, X, Y):
        assert isinstance(self.f, FuncFeatureMap)
        k = self.k
        f = self.f
        assert len(f.output_shape()) == 1
        d_out = f.output_shape()[0]
        n = X.shape[0]
        dx = X[0].numel()
        dy = Y[0].numel()
        assert dx == dy

        # n x d_out x d_out
        kG = k.gradXY_pair(f(X), f(Y))

        g_feat_X = torch.empty((d_out, n, dx),
                               dtype=X.dtype,
                               device=X.device)
        for i_o in range(d_out):
            g_feat_X[i_o] = f.component_grad(X, i_o).reshape(n, dx)

        if Y is X:
            G = torch.einsum('ink, nij, jnk->n', g_feat_X, kG, g_feat_X)
            return G

        g_feat_Y = torch.empty((d_out, n, dy),
                               dtype=Y.dtype,
                               device=Y.device)
        for i_o in range(d_out):
            g_feat_Y[i_o] = f.component_grad(Y, i_o).reshape(n, dy)

        G = torch.einsum('ink, nij, jnk->n', g_feat_X, kG, g_feat_Y)
        return G


class BKGauss(DifferentiableKernel):
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
        Kvec = torch.exp(-D2/(2.0*sigma2))
        return Kvec

    def __str__(self):
        return "KGauss(%.3f)" % self.sigma2

    def gradX(self, X, Y):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a Torch tensor of size nx x ny.
        """
        sigma2 = torch.sqrt(self.sigma2)**2
        K = self.eval(X, Y)
        Diff = X.unsqueeze(1) - Y.unsqueeze(0)
        #Diff = np.reshape(X[:, dim], (-1, 1)) - np.reshape(Y[:, dim], (1, -1))
        G = -torch.einsum('ij,ijk->ijk', K, Diff/sigma2)
        return G

    def gradXY(self, X, Y):
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        sigma2 = torch.sqrt(self.sigma2)**2
        diff = (X.unsqueeze(1) - Y.unsqueeze(0))
        outer = diff.unsqueeze(3) * diff.unsqueeze(2)
        outer = -outer
        idx = torch.arange(d1)
        outer[:, :, idx, idx] += sigma2
        K = self.eval(X, Y)
        G = torch.einsum('ij,ijkl->ijkl', K, outer)/(sigma2**2)
        # print(K[0, 0]/sigma2 * outer[0, 0] - G[0, 0])
        return G

    def gradXY_pair(self, X, Y):
        assert X.shape==Y.shape, 'Two inputs must have the same shape'
        d = X.shape[1]
        sigma2 = torch.sqrt(self.sigma2)**2
        diff = X - Y
        outer = torch.einsum('ij, ik->ijk', diff, diff)
        outer = -outer
        idx = torch.arange(d)
        outer[:, idx, idx] += 1./sigma2
        K = self.pair_eval(X, Y)
        G = torch.einsum('i,ijk->ijk', K, outer)/(sigma2**2)
        return G



class KGauss(BKGauss, KSTKernel):
    def __init__(self, sigma2):
        """
        sigma2: a number representing squared width
        """
        super(KGauss, self).__init__(sigma2)

    def parX(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a Torch tensor of size nx x ny.
        """
        sigma2 = torch.sqrt(self.sigma2)**2
        K = self.eval(X, Y)
        Diff = X[:, [dim]] - Y[:, [dim]].T
        #Diff = np.reshape(X[:, dim], (-1, 1)) - np.reshape(Y[:, dim], (1, -1))
        G = -K*Diff/sigma2
        return G
    
    def parY(self, X, Y, dim):
        return self.parX(Y, X, dim).T

    def gradXY_sum(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\par^2 k(X, Y)}{\par x_i \par y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny Torch tensor of the derivatives.
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        d = d1
        sigma2 = torch.sqrt(self.sigma2)**2
        D2 = torch.sum(X**2, 1).view(n1, 1) - 2*torch.matmul(X, Y.T) + torch.sum(Y**2, 1).view(1, n2)
        K = torch.exp(-D2/(2.0*sigma2))
        G = K/sigma2 * (d - D2/sigma2)
        return G

    def gradXY_sum_pair(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\par^2 k(X, Y)}{\par x \par y}
        evaluated at each x_i in X, and y_j in Y.

        Args:
            X: nx x d numpy array.
            Y: ny x d numpy array.

        Return a nx x ny Torch tensor of the derivatives.
        """
        assert X.shape == Y.shape
        d = X.shape[1]
        sigma2 = torch.sqrt(self.sigma2)**2
        D2 = torch.sum((X-Y)**2, axis=1)
        K = torch.exp(-D2/(2.0*sigma2))
        G = K/sigma2 * (d - D2/sigma2)
        return G


class PTExplicitKernel(Kernel):
    """
    A class for kernel that is defined as 
        k(x,y) = <f(x), f(y)> 
    for a finite-output f (of type FeatureMap).
    """

    def __init__(self, fm):
        """
        fm: a FeatureMap parameterizing the kernel. This feature map is
            expected to take in a Pytorch tensor as the input.
        """
        self.fm = fm

    @abstractmethod
    def eval(self, X, Y):
        """
        Evaluate the kernel on Pytorch tensors X and Y
        X: nx x d where each row represents one point
        Y: ny x d
        return nx x ny Gram matrix
        """
        f = self.fm
        FX = f(X)
        FY = f(Y)
        K = FX.mm(FY.t())
        return K

    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...
        
        X: n x d where each row represents one point
        Y: n x d
        return a 1d Pytorch array of length n.
        """
        f = self.fm
        FX = f(X)
        FY = f(Y)
        vec = torch.sum(FX * FY, 1)
        return vec

    # def is_compatible(self, k):
    #     """
    #     This compatibility check is very weak.
    #     """
    #     if isinstance(k, PTExplicitKernel):
    #         fm1 = self.fm
    #         fm2 = k.fm
    #         return fm1.input_shape() == fm2.input_shape() and \
    #                 fm1.output_shape() == fm2.output_shape()
    #     return False

    def get_feature_map(self):
        return self.fm


class BKIMQ(DifferentiableKernel):
    
    def __init__(self, b=-0.5, c=1.0):
        if not b < 0:
            raise ValueError("b has to be negative. Was {}".format(b))
        if not c > 0:
            raise ValueError("c has to be positive. Was {}".format(c))
        self.b = b
        self.c = c

    def eval(self, X, Y):
        b = self.b
        c = self.c
        D2 = util.pt_dist2_matrix(X, Y)
        K = (c ** 2 + D2) ** b
        return K

    def pair_eval(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        b = self.b
        c = self.c
        return (c ** 2 + torch.sum(((X - Y)) ** 2, 1)) ** b

    def gradX(self, X, Y, shift=-1):
        """
        Compute the gradient with respect to X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny x d
        """
        D2 = util.pt_dist2_matrix(X, Y)
        diff = (X.unsqueeze(1) - Y.unsqueeze(0))

        b = self.b
        c = self.c
        G = torch.einsum('ij,ijk->ijk', ( 2.0*b*(c**2 + D2)**(b-1) ), diff)
        return G

    def gradXY(self, X, Y):
        """
        Compute grad_x grad_y k(x, y)

        X: nx x d torch.Tensor
        Y: ny x d torch.Tensor

        Return a nx x ny x d x d torch tensor of the derivatives.
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        b = self.b
        c = self.c
        diff = (X.unsqueeze(1) - Y.unsqueeze(0))
        outer = diff.unsqueeze(3) * diff.unsqueeze(2)
        D2 = util.pt_dist2_matrix(X, Y)
        # d = input dimension
        c2D2 = c**2 + D2
    
        G = torch.einsum('ij,ijkl->ijkl', -4.0*b*(b-1)*(c2D2**(b-2)), outer)
        diag = -2.0*b*c2D2**(b-1)
        for i in range(d1):
            G[:, :, i, i] += diag
        return G

    def gradXY_pair(self, X, Y):
        """
        Compute pair-wise grad_x grad_y k(x, y)

        X: n x d torch.Tensor
        Y: n x d torch.Tensor

        Return a n x d x d torch tensor of the derivatives.
        """
        assert X.shape == Y.shape, 'Two inputs must have the shape'
        d = X.shape[1]
        b = self.b
        c = self.c
        diff = X - Y
        outer = torch.einsum('ij, ik->ijk', diff, diff)
        D2 = torch.sum((X-Y)**2, 1)
        # d = input dimension
        c2D2 = c**2 + D2

        G = torch.einsum('i,ijk->ijk', -4.0*b*(b-1)*(c2D2**(b-2)), outer)
        diag = -2.0*b*c2D2**(b-1)
        for i in range(d):
            G[:, i, i] += diag
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
        self.s2 = (torch.tensor(s2) if not isinstance(s2, torch.Tensor)
                   else s2)

    def eval(self, X, Y):
        b = self.b
        c = self.c
        s2 = torch.sqrt(self.s2)**2
        D2 = util.pt_dist2_matrix(X, Y)
        K = (c ** 2 + D2/s2) ** b
        return K

    def pair_eval(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        b = self.b
        c = self.c
        s = torch.sqrt(self.s2)
        return (c ** 2 + torch.sum(((X - Y)/s) ** 2, 1)) ** b

    def gradX(self, X, Y, shift=-1):
        d = X.shape[1]
        G = torch.stack([self.parX(X, Y, j) for j in range(d)])
        return G.permute(1, 2, 0)

    def gradX_pair(self, X, Y):
        b = self.b
        c = self.c
        diff = X - Y
        s2 = torch.sqrt(self.s2)**2
        D2 = torch.sum((diff)**2, axis=1)
        G = torch.einsum('ij,i->ij', diff/s2, 2.0*b*(c**2 + D2/s2)**(b-1))
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
        Gdim = ( 2.0*b*(c**2 + D2/s2)**(b-1) )*dim_diff / s2
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
        return T1 + T2

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
        return T1 + T2


class KSTRegularizedMQ(KSTKernel):
    """Class of MultiQuadratic (IMQ) kernels regularized to be positive definite.
    Have not been tested. Be careful. 
    """

    def __init__(self, b=0.5, c=1.0, s2=1.0):
        self.b = b
        self.c = c
        self.s2 = (torch.tensor(s2) if not isinstance(s2, torch.Tensor)
                   else s2)

    def eval(self, X, Y):
        b = self.b
        c = self.c
        s2 = torch.sqrt(self.s2)**2
        D2 = util.pt_dist2_matrix(X, Y)
        DX = util.pt_dist2_matrix(X, torch.zeros_like(Y))
        DY = util.pt_dist2_matrix(torch.zeros_like(X), Y)
        K = (c ** 2 + D2/s2) ** b
        K -= (c ** 2 + DX/s2) ** b
        K -= (c ** 2 + DY/s2) ** b
        K += c ** (2*b) 
        K = (-1)**ceil(b) * K + 1
        return K

    def pair_eval(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        b = self.b
        c = self.c
        s = torch.sqrt(self.s2)

        K = (c ** 2 + torch.sum(((X - Y)/s) ** 2, 1)) ** b
        K -= (c ** 2 + torch.sum(((torch.zeros_like(X) - Y)/s) ** 2, 1)) ** b
        K -= (c ** 2 + torch.sum(((X - torch.zeros_like(Y))/s) ** 2, 1)) ** b
        K += c ** (2*b) 
        K = (-1)**ceil(b) * K + 1
        return K

    def gradX(self, X, Y, shift=-1):
        d = X.shape[1]
        G = torch.stack([self.parX(X, Y, j) for j in range(d)])
        return G.permute(1, 2, 0)

    def parX(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        s2 = torch.sqrt(self.s2)**2
        D2 = util.pt_dist2_matrix(X, Y)
        D2_X = util.pt_dist2_matrix(X, torch.zeros_like(Y))
        # 1d array of length nx
        Xi = X[:, dim]
        # 1d array of length ny
        Yi = Y[:, dim]
        # nx x ny
        dim_diff = (Xi.unsqueeze(1) - Yi.unsqueeze(0))
        assert dim_diff.shape == (X.shape[0], Y.shape[0])

        b = self.b
        c = self.c

        Gdim = ( 2.0*b*(c**2 + D2/s2)**(b-1) )*dim_diff / s2
        Gdim = Gdim - ( 2.0*b*(c**2 + D2_X/s2)**(b-1) )*Xi.unsqueeze(1) / s2
        Gdim = (-1)**ceil(b) * Gdim
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


class KEucNorm(KSTKernel):

    def __init__(self, p=1, c=1, loc=None, scale=1.):
        self.p = p
        self.c = c
        self.loc = loc 
        self.scale = scale

    def eval(self, X, Y):
        p = self.p
        c = self.c 
        s = self.scale 
        loc = self.loc
        if loc is None:
            loc = torch.zeros_like(X[0])
        
        Xnorm = torch.norm((X-loc)/s, 2, dim=-1, keepdim=False)
        Ynorm = torch.norm((Y-loc)/s, 2, dim=-1, keepdim=False)
        return torch.outer(c+Xnorm**p, c+Ynorm**p)

    def pair_eval(self, X, Y):
        p = self.p
        c = self.c 
        s = self.scale 
        loc = self.loc
        if loc is None:
            loc = torch.zeros_like(X[0])
        
        Xnorm = torch.norm((X-loc)/s, 2, dim=-1, keepdim=False)
        Ynorm = torch.norm((Y-loc)/s, 2, dim=-1, keepdim=False)
        return (c+Xnorm**p) * (c+Ynorm**p)

    def parX(self, X, Y, dim):
        p = self.p
        c = self.c
        s = self.scale 
        loc = self.loc
        if loc is None:
            loc = torch.zeros_like(X[0])
  
        Xnorm = torch.norm((X-loc)/s, 2, dim=-1, keepdim=False)
        Ynorm = torch.norm((Y-loc)/s, 2, dim=-1, keepdim=False)
        return p*torch.outer(Xnorm**(p-2) * (X[:, dim]-loc[dim])/s**2, c + Ynorm**p) 
        

    def gradX(self, X, Y):
        d = X.shape[1]
        G = torch.stack([self.parX(X, Y, j) for j in range(d)])
        return G.permute(1, 2, 0)
    

    def gradXY_sum(self, X, Y):
        p = self.p
        c = self.c
        s = self.scale 
        loc = self.loc
        if loc is None:
            loc = torch.zeros_like(X[0])

        Xnorm = torch.norm((X-loc)/s, 2, dim=-1, keepdim=False)
        Ynorm = torch.norm((Y-loc)/s, 2, dim=-1, keepdim=False)

        s2 = s**2
        grad_sum = p**2 * torch.outer(Xnorm**(p-2), Ynorm**(p-2))
        grad_sum = grad_sum * torch.einsum('ij, kl->ik', (X-loc)/s2, (Y-loc)/s2)
        return grad_sum



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


class KSTSumKernel(KSTKernel):
    """Class representing a (weighted) sum of KST kernels

    Attributes:
        kernels:
            A list or a tuple of KSTKernel objects
        weight_tensor: 
            Vector of weights in computing the sum.
            Defaults to (1/T, .... 1/T) if
            there are T kernels
    """
    def __init__(self, kernels, weight_tensor=None):
        self.kernels = kernels
        n_kernels = len(kernels)
        self.n_kernels = n_kernels
        default_w = torch.ones([n_kernels]) / n_kernels
        self.weight_tensor = (weight_tensor if weight_tensor is not None
                              else default_w
                              )
    
    def _sum_op(self, X, Y, method_str,
                shape_tuple, dtype, device):
        ks = self.kernels
        w = self.weight_tensor
        K = torch.zeros(shape_tuple, dtype=dtype,
                        device=device)
        for i, k in enumerate(ks):
            K += w[i] * getattr(k, method_str)(X, Y)
        return K

    def eval(self, X, Y):
        nx, dx = X.shape
        ny, dy = Y.shape
        assert dx == dy

        K = self._sum_op(X, Y, 'eval',
                         [nx, ny],
                         X.dtype, X.device)
        return K
    
    def pair_eval(self, X, Y):
        assert X.shape == Y.shape
        n, d = X.shape
        K = self._sum_op(X, Y, 'pair_eval',
                         [n, ], X.dtype,
                         X.device)
        return K 

    def gradX(self, X, Y):
        nx, dx = X.shape
        ny, dy = Y.shape
        assert dx == dy

        G = self._sum_op(X, Y, 'gradX',
                         [nx, ny, dx],
                         X.dtype, X.device)
        return G
    
    def gradX_pair(self, X, Y):
        assert X.shape == Y.shape
        n, d = X.shape

        G = self._sum_op(X, Y, 'gradX_pair',
                         [n, d],
                         X.dtype, X.device)
        return G

    def gradXY_sum(self, X, Y):
        nx, dx = X.shape
        ny, dy = Y.shape
        assert dx == dy

        G = self._sum_op(X, Y, 'gradXY_sum',
                         [nx, ny],
                         X.dtype, X.device)
        return G

    def gradXY_sum_pair(self, X, Y):
        assert X.shape == Y.shape
        n, d = X.shape

        G = self._sum_op(X, Y, 'gradXY_sum_pair',
                         [n, d],
                         X.dtype, X.device)
        return G
 
class BKLinear(DifferentiableKernel):

    def __init__(self):
        super(BKLinear, self).__init__()
    
    def eval(self, X, Y):
        return X @ Y.T

    def pair_eval(self, X, Y):
        return torch.sum(X * Y, axis=0)

    def gradX(self, X, Y):
        nx, dx = X.shape
        ny, dy = Y.shape
        assert dx == dy
        
        return torch.stack([Y for _ in range(nx)])
    
    def gradXY(self, X, Y):
        nx, dx = X.shape
        ny, dy = Y.shape
        assert dx == dy
        
        G = torch.zeros([nx, nx, dx, dy], dtype=X.dtype,
                        device=X.device)
        idx = torch.arange(dx)
        G[:, :, idx, idx] += 1.
        return G

    def gradXY_pair(self, X, Y):
        assert X.shape == Y.shape
        n, d = X.shape
        
        G = torch.zeros([n, d, d], dtype=X.dtype,
                        device=X.device)
        idx = torch.arange(d)
        G[:, idx, idx] += 1.
        return G



class KSTProduct(KSTKernel):
    """KSTKernel representing the product of two kernel

    Attributes:
        k1:
            KSTKernel object
        k2: 
            KSTKernel object
    """
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def eval(self, X, Y):
        K = self.k1.eval(X, Y)
        K *= self.k2.eval(X, Y)
        return K

    def pair_eval(self, X, Y):
        K = self.k1.pair_eval(X, Y)
        K *= self.k2.pair_eval(X, Y)
        return K

    def gradX(self, X, Y):
        k1 = self.k1
        k2 = self.k2 
        K1 = k1.eval(X, Y)
        K2 = k2.eval(X, Y)
        G1 = k1.gradX(X, Y)
        G2 = k2.gradX(X, Y)
        T1 = torch.einsum('ij,ijk->ijk', K1, G2)
        T2 = torch.einsum('ij,ijk->ijk', K2, G1)
        return T1 + T2

    def gradX_pair(self, X, Y):
        k1 = self.k1
        k2 = self.k2 
        K1 = k1.pair_eval(X, Y)
        K2 = k2.pair_eval(X, Y)
        G1 = k1.gradX_pair(X, Y)
        G2 = k2.gradX_pair(X, Y)
        return K1 * G2 + K2 * G2

    def gradXY_sum(self, X, Y):
        k1 = self.k1
        k2 = self.k2
        K1 = k1.eval(X, Y)
        K2 = k2.eval(X, Y)
        T1 = K1 * k2.gradXY_sum(X, Y) + K2*k1.gradXY_sum(X, Y)
        T2 = torch.einsum('ijk,ijk->ij', k1.gradX(X, Y), k2.gradY(X, Y))
        T2 += torch.einsum('ijk,ijk->ij', k2.gradX(X, Y), k1.gradY(X, Y))
        return T1 + T2

    def gradXY_sum_pair(self, X, Y):
        k1 = self.k1
        k2 = self.k2
        K1 = k1.pair_eval(X, Y)
        K2 = k2.pair_eval(X, Y)
        T1 = K1 * k2.gradXY_sum_pair(X, Y) + K2*k1.gradXY_sum_pair(X, Y)
        T2 = k1.gradX_pair(X, Y) * k2.gradX_pair(Y, X)
        T2 += k2.gradX_pair(X, Y) * k1.gradX_pair(Y, X)
        return T1 + T2


kernel_derivatives = {
    BKGauss: lambda k: 1.,
    BKIMQ: lambda k: -2*k.b*(k.c**2)**(k.b-1),
    BKLinear: lambda k: 1.,
}

import torch
import torch.nn as nn


class TwoLayerFC(nn.Module):
    
    def __init__(self, din, dh1, dh2, dout,
                 activation=nn.ReLU):
        super(TwoLayerFC, self).__init__()
        self.din = din
        self.dh1 = dh1
        self.dh2 = dh2
        self.dout = dout
        self.activation = activation
        self.module = nn.Sequential(
           nn.Linear(din, dh1),
           activation(),
           nn.Linear(dh1, dh2),
           activation(),
           nn.Linear(dh2, dout),
        )
        
    def forward(self, X):
        return self.module(X)


class MultipleLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, n_linears: int,
                 bias=True
                 ):
        super(MultipleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_linears = n_linears
        self.weight = nn.Parameter(
            torch.Tensor(in_features, n_linears, out_features))
        self.weight = nn.init.normal_(self.weight) 
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_linears, out_features))
            self.bias = nn.init.zeros_(self.bias)
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        
    def forward(self, X):
        W = self.weight
        b = (0. if self.bias is None
             else self.bias)
        if len(X.shape) == 2:
            Y = torch.einsum('bi,ijk->bjk', X, W) + b
            return Y
        return torch.einsum('nbi, ijk->nbjk', X, W) + b

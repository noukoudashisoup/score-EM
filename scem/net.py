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

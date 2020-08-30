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

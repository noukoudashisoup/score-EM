import sys, os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

import torch
import torch.nn as nn
from torch.distributions import Normal, MixtureSameFamily, Categorical, Independent, transforms, StudentT, Laplace
import torch.distributions as td
import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm
from scem import ebm, stein, kernel, util, gen

import argparse

device = torch.device("cuda:0")

def score(p, x):
    x = x.clone().requires_grad_()
    s = torch.autograd.grad(p.log_prob(x).sum(), [x])[0]
    return s

class SLaplace(Laplace):
    
    def score(self, x):
        return -torch.sign(x - self.loc) / self.scale

class SStudentT(StudentT):
    
    def score(self, x):
        d = self.df
        m  = self.loc
        s  = self.scale
        z = (x - m)/s
        return - (d + 1) * z / (d + z**2) / s

class SNormal(Normal):
    
    def score(self, x):
        
        return -( x - self.mean ) / self.scale**2
    
    def div_score(self, x):
        
        return - 1. / self.scale**2 * torch.ones_like(x).sum(-1)
        

class SGMM(MixtureSameFamily):
    
    def score(self, x):
        
        logpx_z = self.component_distribution.log_prob(x[...,None,:])
        logpxz  = logpx_z + self.mixture_distribution.probs.log()
        logpx   = torch.logsumexp(logpxz, -1)
        
        pz_x = (logpxz - logpx[...,None]).exp()

        dxlogpx_z = self.component_distribution.base_dist.score(x[...,None,:])
        s = (dxlogpx_z * pz_x[...,None]).sum(-2)

        return s
    

def get_dist(dist, pi, mean):
    if dist == "N":
        Dist = SNormal
    elif dist == "L":
        Dist = SLaplace
    elif dist == "S":
        Dist = lambda m, s: SStudentT(5, m, s)

    q = Dist(torch.tensor([mean], device=device),torch.ones(1, device=device)*1)
    mixing = Categorical(probs = torch.tensor([pi,1-pi], device=device))
    component = Independent(Dist(torch.tensor([-5,5.0], device=device)[:,None],torch.ones(1,1, device=device)), 1)
    p = SGMM(mixing, component, 1)
    return q, p


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pi", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dist", type=str, default="S")
    parser.add_argument("--init_noise", type=float, default=5.0)
    parser.add_argument("--noise_interval", type=int, default=500)
    parser.add_argument("--mean", type=float, default=5)
    parser.add_argument("--inner_steps", type=int, default=10000)
    parser.add_argument("--alg", type=str, default="svgd", choices=["svgd", "spos", "lang"])
    parser.add_argument("--noise_type", type=str, default="decreasing", choices=["decreasing", "fixed"])
    args = parser.parse_args()

    pi = args.pi
    sigma = args.sigma
    seed = args.seed
    dist = args.dist
    init_noise = args.init_noise
    noise_interval = args.noise_interval
    mean = args.mean
    inner_steps = args.inner_steps
    alg = args.alg
    noise_type = args.noise_type

    if ((alg == "svgd" and noise_type == "decreasing") or (alg == "lang" and noise_type == "fixed")):
        exit()

    fn = f"D{dist}_pi{pi}_sigma{sigma}_noise{init_noise:.1f}_mean{mean}_is{inner_steps}_{alg}_nt{noise_type}_seed{seed:02d}"

    if os.path.exists(f'results/{fn}.npz'):
        exit()

    print(fn)
    
    q, p = get_dist(dist, pi, mean)


    torch.random.manual_seed(seed)
    kx = kernel.KGauss(torch.tensor([sigma], device=device))

    def stein_grad(kx, samples, s, x, step_size=1):
        return step_size * (s(samples)[:,None,:] * kx.eval(samples, x)[...,None] + kx.gradX(samples, x)).mean(0)


    def stochastic_stein_grad(kx, samples, s, x, step_size=1, noise = 1):
        score = s(samples)
        sg = step_size * (score[:,None,:] * kx.eval(samples, x)[...,None] + kx.gradX(samples, x)).mean(0)
        lg = score * step_size * noise + torch.sqrt(2 * noise * step_size) * torch.randn_like(x)
        return sg + lg

    def langevin_grad(kx, samples, s, x, step_size=1, noise = 1):
        lg = s(x) * step_size * noise + torch.sqrt(2 * noise * step_size) * torch.randn_like(x)
        return lg
                    

    x = q.sample([1000]).to(device)

    nsteps = int(100)
    inner_steps = int(args.inner_steps)

    props = np.zeros(nsteps*inner_steps)
    stds = np.zeros(nsteps*inner_steps)
    stein_grads = np.zeros(nsteps*inner_steps)
    if noise_type == "fixed" and alg != 'spos':
        noises = torch.linspace(init_noise, init_noise, nsteps)
    else:
        noises = torch.linspace(init_noise, 0, nsteps)


    counter = 0

    for noise in tqdm.tqdm(noises):

        for i in range(inner_steps):

            if alg == "svgd":
                grad_x = stein_grad(kx, x, p.score, x, step_size = 1)
            elif alg == "spos" and noise_type == "fixed":
                grad_x = stochastic_stein_grad(kx, x, p.score, x, step_size = noise, noise=1.0)
            elif alg == "spos" and noise_type == "decreasing":
                grad_x = stochastic_stein_grad(kx, x, p.score, x, step_size = 1, noise=noise)
            elif alg == "lang":
                grad_x = langevin_grad(kx, x, p.score, x, step_size = 1, noise=noise)

            x = x + 1 * grad_x
            stein_grads[counter] = grad_x.norm(dim=-1).mean().item()
            props[counter] = (x>0).to(torch.float).mean().item()
            stds[counter]= x.std().item()
            counter += 1
        
        
    np.savez("results/"+fn+".npz", props = props, stein_grads = stein_grads, x=x.cpu().numpy(), stds = stds)

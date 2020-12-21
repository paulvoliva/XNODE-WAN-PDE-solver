# -*- coding: utf-8 -*-
"""PDE Solver.ipynb
Here we will take the time components in the third axis
"""

import math
import torch
import signatory
import LOGSIG
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from Earlystop import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

"""# Exact solution u(x) for the example PDE
we conduct a experiment on solving a IBVP with nonlinear diffusion-reaction equation and boundary condition involving time:
\begin{equation}
\left\{\begin{array}{ll}
u_{t}-\Delta u-u^{2}=f(x, y, t), & \text { in } \Omega \times[0, T] \\
u(x, y, t)=g(x, y, t), & \text { on } \partial \Omega \times[0, T] \\
u(x, y, 0)=h(x, y), & \text { in } \Omega
\end{array}\right.
\end{equation}
where $\Omega=(-1,1)^{2} \subset \mathbb{R}^{2}$. In this test, we give the definition of $f(x,y,t)=\left(\pi^{2}-2\right) \sin \left(\frac{\pi}{2} x\right) \cos \left(\frac{\pi}{2} y\right) e^{-t} - 4 \sin ^{2}\left(\frac{\pi}{2} x\right) \cos \left(\frac{\pi}{2} y\right) e^{-2 t}$ in $\Omega \times[0, T]$, $g(x,y,t)=2 \sin \left(\frac{\pi}{2} x\right) \cos \left(\frac{\pi}{2} y\right) e^{-t}$ on $\partial \Omega \times[0, T]$ and $h(x,y)=2 \sin \left(\frac{\pi}{2} x\right) \cos \left(\frac{\pi}{2} y\right)$ in $\Omega$. And the true solution is $u(x,y,t)=2 \sin \left(\frac{\pi}{2} x\right) \cos \left(\frac{\pi}{2} y\right) e^{-t}$.

# PDE Setup
"""

#'''
# I use this to get the shape of the tensor when I debug
old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info
#'''

def func_u_sol(xt):
    l = xt.shape[0]
    u = 2 * torch.sin(math.pi / 2 * xt[:, 0, :]) * torch.cos(math.pi / 2 * xt[:, 1, :]) * torch.exp(xt[:, 2, :])
    return (u)


# We denote spatial coordinates with time as 'xt' and 'x' without

def func_f(xt):
    l = xt.shape[0]
    f = (math.pi ** 2 - 2) * torch.sin(math.pi / 2 * xt[:, 0, :]) * torch.cos(math.pi / 2 * xt[:, 1, :]) * torch.exp(
        -xt[:, 2, :]) - 4 * torch.sin(math.pi / 2 * xt[:, 0, :]) ** 2 * torch.cos(math.pi / 2 * xt[:, 1, :]) * torch.exp(-xt[:, 2, :])
    return (f)


def func_g(boundary_xt):
    return func_u_sol(boundary_xt)


def func_h(x):
    h = 2 * torch.sin(math.pi / 2 * x[:, 0]) * torch.cos(math.pi / 2 * x[:, 1])
    return h


def func_w(x):  # returns 1 for positions in the domain and 0 otherwise
    lens = x.shape[0]
    w_bool = torch.gt(1 - torch.abs(x[:, 0]), torch.zeros(lens)) & torch.gt(torch.abs(x[:, 0]), torch.zeros(lens))
    w_val = torch.where(w_bool, 1 - torch.abs(x[:, 0]) + torch.abs(x[:, 0]), torch.zeros(lens))
    return (w_val.view(-1, 1))


"""# Domain"""
# TODO: Understand the input format
T0 = 0  # if this is ignored it is always set as T0=0
T = 1

# Set up for a square
up = 1.0
down = -1.0
dim = 2
t_mesh_size = 100
r_mesh_size = 50

# defining the training domain
x_domain = torch.Tensor(dim, t_mesh_size).uniform_(down, up)
t = torch.linspace(0, 1, t_mesh_size).unsqueeze(1).view(1, t_mesh_size)
xt_domain = torch.cat((x_domain, t), dim=0)
xt_domain.requires_grad_()
r_domain = torch.Tensor(r_mesh_size).uniform_(T0, T)
r_domain = r_domain[r_domain.sort()[1]].unsqueeze(1).repeat(1,3)
r_domain[0], r_domain[-1] = T0, T

def X(seq, t):
    t0 = min(seq[2, :].data, key=lambda x:abs(x-t))
    idx1 = torch.nonzero((seq[2, :]==t0)).item()
    if t-t0==0:
        xyt = torch.tensor([seq[0, idx1], seq[1, idx1], seq[2, idx1]])
    else:
        factor = ((t-t0)/abs(t-t0)).item()
        idx2 = int(idx1+factor)
        xyt = torch.tensor([seq[0, idx1]+(t-t0)*torch.abs(seq[0, idx2]-seq[0, idx1]), seq[1, idx1]+(t-t0)*torch.abs(seq[1, idx2]-seq[1, idx1]), t])
    return xyt

for i in range(r_mesh_size):
    r_domain[i,:] = X(xt_domain, r_domain[i,0])

r_domain.requires_grad_(True)

"""# Defining the Model"""

class Func_f(torch.nn.Module):
    def __init__(self, input_dim, logsig_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.logsig_dim = logsig_dim
        self.num_layers = config['layers_f']
        self.dim = config['dim_f']
        additional_layers = [torch.nn.ReLU(), torch.nn.Linear(dim, dim)]*(self.num_layers-1) if self.num_layers > 1 else []
        self.net = torch.nn.Sequential(
            *[torch.nn.Linear(input_dim, dim),
            *additional_layers,
            torch.nn.Tanh(),
            torch.nn.Linear(dim, input_dim*logsig_dim),]
        )

    def forward(self, x):
        return self.net(h).view(-1, self.input_dim, self.logsig_dim)

class generator(torch.nn.Module):
    def __init__(self, input_dim, logsig_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.logsig_dim = logsig_dim
        self.dim = config['u_hidden_dim']

        self.initial_lin = torch.nn.Linear(input_dim, self.dim)
        self.func_f = Func_f(self.dim, logsig_dim, config)
        self.final_lin = torch.nn.Linear(self.dim, 1)

    def forward(self, initial, logsig):
        h0 = self.initial_lin(initial)
        h = LOGSIG.rdeint(logsig, h0, self.func_f, method='midpoint', adjoint=False, return_sequences=True)
        out = self.final_lin(h)
        return out

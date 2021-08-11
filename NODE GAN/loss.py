"""
loss.py
===========================
This is the loss function for the model
"""

import torch
from itertools import product
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class loss:
    '''
    This is the loss class; in it all the various losses are calculated and we can call the losses for the generator and
    discriminator within the class (here denoted by `u` and `v` respectively). The function `I` calculates
    $\langle A[u_\theta], v_\phi \rangle$. The function `int` represents the interior loss, `init` the intial loss (at
    time T0) and `bdry` the loss on the boundary.

    Args:
        alpha: the parameter that assigns relative weights to the initial and boundary loss
        a: Tensor with the values of $a$ (from the general form) evaluated at the points of the domain
        b: Tensor with the values of $b$ (from the general form) evaluated at the points of the domain
        h: Tensor with the values of the function $h$ (from the general form) evaluated at the points of the domain
        f: Tensor with the values of the function $f$ (from the general form) evaluated at the points of the domain
        g: Tensor with the values of the function $g$ (from the general form) evaluated at the points of the domain
        domain: the domain object (see dataset)
    '''

    def __init__(self, alpha: float, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, h: torch.Tensor, f: torch.Tensor,
                 g: torch.Tensor, setup: torch.Tensor, domain):
        super(loss).__init__()
        self.T = setup['T']
        self.T0 = setup['T0']
        self.alpha = alpha
        self.a = a
        self.b = b
        self.h = h
        self.f = f
        self.g = g
        self.setup = setup
        self.c = c
        self.func_w = domain.func_w
        self.V = domain.V()

    def I(self, y_output_u: torch.Tensor, y_output_v: torch.Tensor, X: torch.Tensor, XV: torch.Tensor):
        y_output_u.retain_grad()
        y_output_v.retain_grad()
        N = y_output_u.shape[0]
        xw = XV.clone().detach()
        w = self.func_w(xw).unsqueeze(2).to(device)
        phi = y_output_v * w
        X.retain_grad()
        XV.retain_grad()
        y_output_u.backward(torch.ones_like(y_output_u).to(device), retain_graph=True)
        du = {}
        for i in range(self.setup['dim']):
            du['du_' + str(i)] = X.grad.to(device)[:, :, i]
        phi.backward(torch.ones_like(phi).to(device), retain_graph=True)
        dphi = {}
        for i in range(self.setup['dim'] + 1):
            dphi['dphi_' + str(i)] = XV.grad.to(device)[:, :, i]
        s1 = self.V * (y_output_u[:, -1].squeeze() * y_output_v[:, -1].squeeze() - self.h * y_output_v[:, 0].squeeze()) / N
        s2 = (self.T - self.T0) * self.V * (y_output_u.squeeze() * dphi['dphi_2']) / N
        s31 = [self.a[i, j] * dphi['dphi_' + str(i)] * du['du_' + str(j)] for i, j in
               product(range(self.setup['dim']), repeat=2)]
        s31 = torch.stack(s31, 0).sum(0)
        s32 = np.sum([self.b[i] * phi.squeeze() * du['du_' + str(i)] for i in range(self.setup['dim'])])
        s3f = s31 + s32 + self.c.squeeze() * y_output_u.squeeze() * phi.squeeze() + self.f * phi.squeeze()
        s3c = (self.T - self.T0) * self.V / N
        s3 = s3c * s3f
        I = torch.sum(s1 - torch.sum(s2 - s3, 1), 0)
        X.grad.data.zero_()
        XV.grad.data.zero_()
        return I

    def init(self, y_output_u: torch.Tensor):
        r = torch.mean((y_output_u[:, 0] - self.h.unsqueeze(1)) ** 2)
        return r

    # initially had all variables feed with grad
    def bdry(self, u_net: torch.nn.Module, border_data: torch.Tensor):
        r = torch.mean((u_net(border_data) - self.g.unsqueeze(2)) ** 2)
        return r

    def int(self, y_output_u: torch.Tensor, y_output_v: torch.Tensor, X: torch.Tensor, XV: torch.Tensor):
        # x needs to be the set of points set plugged into net_u and net_v
        N = y_output_v.shape[0]
        return torch.log(self.I(y_output_u, y_output_v, X, XV) ** 2) - torch.log(
            (self.T - self.T0) * self.V * torch.sum(y_output_v ** 2) / N)

    def u(self, y_output_u: torch.Tensor, y_output_v: torch.Tensor, u_net: torch.nn.Module, X: torch.Tensor, XV: torch.Tensor, border: torch.Tensor):
        return self.int(y_output_u, y_output_v, X, XV) + torch.mul((self.init(y_output_u) + self.bdry(u_net, border)), self.alpha)

    def v(self, y_output_u: torch.Tensor, y_output_v: torch.Tensor, X: torch.Tensor, XV: torch.Tensor):
        return torch.mul(self.int(y_output_u, y_output_v, X, XV), -1)

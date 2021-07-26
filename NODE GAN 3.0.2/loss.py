"""
loss.py
===========================
This is the loss function for the model.
"""

import torch
from itertools import product
import numpy as np
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter()

def func_w(x):
    # again we assume that the domain is a hypercube and specifically that the boundaries are 1 and -1

    disttop = torch.min(torch.abs(1-x), dim=2).values
    distbot = torch.min(torch.abs(-1-x), dim=2).values

    dist = torch.minimum(disttop, distbot)

    return dist


# TODO: get loss to its own file and make sure it works

class loss:
    '''
    This is the loss class; in it all the various losses are calculated and we can call the losses for the generator and
    discriminator within the class (here denoted by `u` and `v` respectively). The function `I` calculates
    $\langle A[u_\theta], v_\phi \rangle$. The function `int` represents the interior loss, `init` the intial loss (at
    time T0) and `bdry` the loss on the boundary. (Note that in this non-linear example the function c is defined not as
    a Tensor but a function as it depends on the networks guesses of the function $u_\theta$.

    alpha (int): the $\alpha=\gamma$ parameter that assigns relative weights to the initial and boundary loss
    a (Tensor): Tensor with the values of $a$ (from the general form) evaluated at the points of the domain
    b (Tensor): Tensor with the values of $b$ (from the general form) evaluated at the points of the domain
    h (Tensor): Tensor with the values of the function $h$ (from the general form) evaluated at the points of the domain
    f (Tensor): Tensor with the values of the function $f$ (from the general form) evaluated at the points of the domain
    g (Tensor): Tensor with the values of the function $g$ (from the general form) evaluated at the points of the domain
    setup (dict): dictionary with all the configurations of meshes and the problem dimension
    '''

    def __init__(self, alpha, a, b, h, f, g, setup, c, T=1, T0=0):
        super(loss).__init__()
        self.T = T
        self.T0 = T0
        self.alpha = alpha
        self.a = a
        self.b = b
        self.h = h
        self.f = f
        self.g = g
        self.setup = setup
        self.c = c
        self.V = 2 ** setup['dim']  # Volume of \Omega

    def I(self, y_output_u, y_output_v, ind, X, XV):
        X.retain_grad()
        XV.retain_grad()
        N = y_output_u.shape[0]
        xw = XV
        w = func_w(xw).unsqueeze(2).to(device)
        phi = y_output_v * w
        y_output_u.backward(torch.ones_like(y_output_u).to(device), retain_graph=True)
        du = {}
        for i in range(self.setup['dim'] + 1):
            du['du_' + str(i)] = X.grad[ind*N:(ind+1)*N].to(device)[:, :, i]
        phi.backward(torch.ones_like(phi).to(device), retain_graph=True)
        dphi = {}
        for i in range(self.setup['dim'] + 1):
            dphi['dphi_' + str(i)] = XV.grad[ind*N:(ind+1)*N].to(device)[:, :, i]
        s1 = self.V * (y_output_u[:, -1] * y_output_v[:, -1] - self.h[ind * N:(ind + 1) * N].unsqueeze(1) * y_output_v[:, 0]) / N
        s2 = (self.T - self.T0) * self.V * (y_output_u.squeeze(2) * dphi['dphi_0']) / N
        s31 = [self.a[i, j, ind * N:(ind + 1) * N] * dphi['dphi_' + str(i+1)] * du['du_' + str(j+1)] for i, j in
               product(range(self.setup['dim']), repeat=2)]
        s31 = np.sum(s31)
        s32 = np.sum([self.b[i, ind * N:(ind + 1) * N] * phi.squeeze(2) * du['du_' + str(i+1)] for i in range(self.setup['dim'])])
        c = self.c(y_output_u).squeeze(2)
        s3f = s31 + s32 + c * y_output_u.squeeze(2) * phi.squeeze(2) + self.f[ind * N:(ind + 1) * N] * phi.squeeze(2)
        s3c = (self.T - self.T0) * self.V / N
        s3 = s3c * s3f
        I = torch.sum(s1.squeeze() - torch.sum(s2 - s3, 1), 0)
        X.grad.data.zero_()
        XV.grad.data.zero_()
        return I

    def init(self, y_output_u, ind):
        N = y_output_u.shape[0]
        r = torch.mean((y_output_u[:, 0, :] - self.h[ind * N:(ind + 1) * N].unsqueeze(1)) ** 2)
        writer.add_scalar('init', r.item())
        return r

    # initially had all variables feed with grad
    def bdry(self, ind, u_net, N, border_data, func_g):
        r = torch.mean((u_net(border_data) - func_g(border_data).unsqueeze(2)) ** 2)
        writer.add_scalar('bdry', r.item())
        return r

    def int(self, y_output_u, y_output_v, ind, X, XV):
        # x needs to be the set of points set plugged into net_u and net_v
        N = y_output_v.shape[0]
        return torch.log(self.I(y_output_u, y_output_v, ind, X, XV) ** 2) - torch.log(
            (self.T - self.T0) * self.V * torch.sum(y_output_v ** 2) / N)

    def u(self, y_output_u, y_output_v, u_net, X, XV, border, func_g, ind=0):
        N = y_output_u.shape[0]
        return self.int(y_output_u, y_output_v, ind, X, XV) + torch.mul((self.init(y_output_u, ind) + self.bdry(ind, u_net, N, border, func_g)), self.alpha)

    def v(self, y_output_u, y_output_v, X, XV, ind=0):
        return torch.mul(self.int(y_output_u, y_output_v, ind, X, XV), -1)

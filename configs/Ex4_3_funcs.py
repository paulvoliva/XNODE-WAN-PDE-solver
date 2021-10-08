import math
import torch
from NODE_GAN.main import params
from utils.auxillary_funcs import rel_err

def func_u_sol(X):
    sins = 1
    for i in range(params['dim']):
        sins *= torch.sin(math.pi/2 * X[:, :, i+1] + math.pi/2 * i)
    return (2/math.pi) ** (-params['dim']) * 2 * sins * torch.exp(-X[:, :, 0])


def func_f(X):
    sins = 1
    for i in range(params['dim']):
        sins *= torch.sin(math.pi / 2 * X[:, :, i + 1] + math.pi / 2 * i)
    return (2/math.pi) ** (-params['dim']) * (math.pi ** 2 - 2) * sins * torch.exp(-X[:, :, 0]) - 4 * sins ** 2 * torch.exp(-2*X[:, :, 0])


def func_g(BX):
    return func_u_sol(BX)


def func_h(X):
    sins = 1
    for i in range(params['dim']):
        sins *= torch.sin(math.pi / 2 * X[:, i + 1] + math.pi / 2 * i)
    return (2/math.pi) ** (-params['dim']) * 2 * sins


def func_a(X, i, j):
    if i == j:
        return torch.ones(X.shape[:-1])
    else:
        return torch.zeros(X.shape[:-1])


def func_b(X, i):
    return torch.zeros(X.shape[:-1])


# the following function can take into account the function u, so they have the input `y_output_u` which will be our
# guess solution

def func_c(X, y_output_u):
    return -y_output_u

def stop(self, points, domain):
    return True if rel_err(points, self.u_net, self.func_u_sol, self.p, domain.V(), self.params['N_r'])<0.01 else False

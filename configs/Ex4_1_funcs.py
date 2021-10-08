import math
import torch
from utils.auxillary_funcs import rel_err

def func_u_sol(X):
    return 2 * torch.sin(math.pi / 2 * X[:, :, 1]) * torch.cos(math.pi / 2 * X[:, :, 2]) * torch.exp(-X[:, :, 0])


def func_f(X):
    sincos = torch.sin(math.pi / 2 * X[:, :, 1]) * torch.cos(math.pi / 2 * X[:, :, 2])
    return (math.pi ** 2 - 2) * sincos * torch.exp(-X[:, :, 0]) - 4 * sincos ** 2 * torch.exp(-2*X[:, :, 0])


def func_g(BX):
    return func_u_sol(BX)


def func_h(X):
    return 2 * torch.sin(math.pi / 2 * X[:, 1]) * torch.cos(math.pi / 2 * X[:, 2])


def func_a(X, i, j):
    if i == j:
        return torch.ones(X.shape[:-1])
    else:
        return torch.zeros(X.shape[:-1])


def func_b(X, i):
    return torch.zeros(X.shape[:-1])


def func_c(X, y_output_u):
    return -y_output_u

def stop(self, points, domain):
    return True if rel_err(points, self.u_net, self.func_u_sol, self.p, domain.V(), self.params['N_r'])<0.01 else False

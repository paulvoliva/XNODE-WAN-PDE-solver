from src.training import NODE_WAN_solver
import torch
from src.dataset import *

'''
This is the document that allows to solve the PDEs and to interact with the algorithm.
'''

'''
# General Form of our problem:
\begin{equation}
\left\{\begin{array}{ll}
u_{t}-\sum_{i=1}^d \partial_i (\sum_{j=1}^d a_{ij} \partial_j u) + \sum_{i=1}^d b_i \partial_i u + cu = f, & \text { in } \Omega \times[T0, T] \\
u(x, t)=g(x, t), & \text { on } \partial \Omega \times[T0, T] \\
u(x, 0)=h(x), & \text { in } \Omega
\end{array}\right.
\end{equation}
where $x$ is a d-dimensional vector. You specify these functions just below
'''

# setting to cuda

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


''' # Setting Parameters '''

# dictionary with all the configurations of meshes and the problem dimension

setup = {
    'dim': 5,       # int: dimension of the problem
    'N_t': 20,      # int: number of time samples
    'N_r': 400,     # int: number of spatial samples in the interior
    'N_b': 400,     # int: number of spatial samples on the boundary
    'T0': 0,        # float: initial time
    'T': 1,          # float: final time
    'shape_param': 1
}

# hyperparameters

config = {
    'alpha': 1e4 * 400 * 25,    # float: the coefficient in our loss function
    'u_layers': 8,              # int: depth of the hidden field F
    'u_hidden_dim': 20,         # int: dimensionality of the initial and final layers
    'u_hidden_hidden_dim': 10,  # int: dimensionality of the hidden field F
    'v_layers': 9,              # int: the depth of the adversarial DNN
    'v_hidden_dim': 50,         # int: the dimensionality of the adversarial DNN
    'n1': 2,                    # int: sub-iterations for fitting of the guess
    'n2': 1,                    # int: sub-iterations for fitting of the test function
    'u_rate': 0.015,            # float: learning rate of the guess
    'v_rate': 0.04,             # float: learning rate of the test function
    'min_steps': 5,             # int: smallest number of time steps used in the ODE solver
    'adjoint': False,           # bool: whether to use an adjoint solver for the ODE solver
    'solver': 'midpoint'        # str: the solver to be used in the ODE solver
}

iterations = 2e4+1


'''
# Setting the specific problem to solve
'''


def func_u_sol(X):
    return X[:, :, 0]


def func_f(X):
    return X[:, :, 0]


def func_g(BX):
    return BX[:, :, 0]


def func_h(X):
    return X[:, 0]


def func_a(X, i, j):
    return X[:, :, 0]


def func_b(X, i):
    return X[:, :, 0]


# the following function can take into account the function u, so they have the input `y_output_u` which will be our
# guess solution

def func_c(X, y_output_u):
    return X[:, :, 0]


if __name__ == '__main__':
    params = {**config, **setup, **{'iterations': int(iterations)}}
    solver = NODE_WAN_solver(params, func_a, func_b, func_c, func_h, func_f, func_g, NSphere_TCone)
    solver.train(report=True)

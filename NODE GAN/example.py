from training import NODE_solver
import torch
import math
from dataset import NSphere_Tcone, Hypercube


'''
This is the document that allows to solve the PDEs and to interact with the algorithm. To change the domain to anything
but a hypercube (whose bottom and top side can be input in `Comb_loader`) one needs to modify the data formation in 
`Comb_loader`.
'''

'''
# General Form of our problem:
\begin{equation}
\left\{\begin{array}{ll}
u_{t}-\sum_{i=1}^d \partial_i (\sum_{j=1}^d a_{ij} \partial_j u) + \sum_{i=1}^d b_i \partial_i u + cu = f, & \text { in } \Omega \times[0, T] \\
u(x, t)=g(x, t), & \text { on } \partial \Omega \times[0, T] \\
u(x, 0)=h(x), & \text { in } \Omega
\end{array}\right.
\end{equation}
where $x$ is a d-dimensional vector
'''

# setting to cuda


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
# Setting the specific problem to solve
'''

#He. The input X: torch tensor with size [? ? ?].
def func_u_sol(X):
    return 2 * torch.sin(math.pi / 2 * X[:, :, 1]) * torch.cos(math.pi / 2 * X[:, :, 2]) * torch.exp(-X[:, :, 0])


def func_f(X):
    sincos = torch.sin(math.pi / 2 * X[:, :, 1]) * torch.cos(math.pi / 2 * X[:, :, 2])
    return (math.pi ** 2 - 2) * sincos * torch.exp(-X[:, :, 0]) - 4 * sincos ** 2 * torch.exp(-2*X[:, :, 0])


def func_g(BX):
    return func_u_sol(BX)

#He. The input X: torch tensor with size [? ?].
def func_h(X):
    return 2 * torch.sin(math.pi / 2 * X[:, 1]) * torch.cos(math.pi / 2 * X[:, 2])


def func_a(X, i, j):
    if i == j:
        return torch.ones(X.shape[:-1])
    else:
        return torch.zeros(X.shape[:-1])


def func_b(X, i):
    return torch.zeros(X.shape[:-1])


# the following function can take into account the function u, so they have the input `y_output_u` which will be our
# guess solution
#He. comments not clear to me.

def func_c(X, y_output_u):
    return -y_output_u


''' # Setting Parameters '''

# dictionary with all the configurations of meshes and the problem dimension

setup = {
    'dim': 5, #He. spacial dimension
    'N_t': 20, #He. number of points randomly sampled in the time domain [T0, T]?
    'N_r': 400, #He. number of points randomly sampled in the interiori domain $D = \Omega \times[0, T]$.
    'N_b': 400, #He. number of points randomly sampled in the domain boundaty $\partial D := \partial \Omega \times[0, T]$
    'T0': 0, #He. Initial time
    'T': 1 #He. Final (terminal) time
}

# Hyperparameters
#He. Add the explaination for each hyperparameter below.
config = {
    'alpha': 1e4 * 400 * 25,    # float
    'u_layers': 8,              # int
    'u_hidden_dim': 20,         # int
    'u_hidden_hidden_dim': 10,  # int
    'v_layers': 9,              # int
    'v_hidden_dim': 50,         # int
    'n1': 2,                    # int
    'n2': 1,                    # int
    'u_rate': 0.015,            # float
    'v_rate': 0.04,             # float
    'min_steps': 5,             # int
    'adjoint': False,           # bool
    'solver': 'midpoint'        # str
}

iterations = 201

if __name__ == '__main__':
    params = {**config, **setup, **{'iterations': int(iterations)}}
    solver = NODE_solver(params, func_a, func_b, func_c, func_h, func_f, func_g,
                         Hypercube(1, -1, setup['dim'], setup['T0'], setup['T'], setup['N_t'])
                         )
    solver.train(report=True)

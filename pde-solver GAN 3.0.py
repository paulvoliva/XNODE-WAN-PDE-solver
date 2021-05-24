import math
import torch
import os
from ray import tune
from torch.utils.data import DataLoader, Dataset
from ray.tune.schedulers import ASHAScheduler
from torch.autograd import grad
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
# from Earlystop import EarlyStopping
from itertools import product
from torch.utils.tensorboard import SummaryWriter

# Here we set up some of the packages we will use
writer = SummaryWriter()
# torch.multiprocessing.set_start_method('spawn')

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

'''
# I use this to get the shape of the tensor when I debug
old_repr = torch.Tensor.__repr__

def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)

torch.Tensor.__repr__ = tensor_info
# '''

# setting to cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
# Setting the specific problem to solve
'''


# We denote spatial coordinates with time as 'xt' and 'x' without

def func_u_sol(xt):
    l = xt.shape[0]
    u = 2 * torch.sin(math.pi / 2 * xt[:, 0, :]) * torch.cos(math.pi / 2 * xt[:, 1, :]) * torch.exp(-xt[:, 2, :])
    return (u)


def func_f(xt):
    l = xt.shape[0]
    f = (math.pi ** 2 - 2) * torch.sin(math.pi / 2 * xt[:, 0, :]) * torch.cos(math.pi / 2 * xt[:, 1, :]) * torch.exp(
        -xt[:, 2, :]) - 4 * torch.sin(math.pi / 2 * xt[:, 0, :]) ** 2 * torch.cos(
        math.pi / 2 * xt[:, 1, :]) ** 2 * torch.exp(-2 * xt[:, 2, :])
    return (f)


def func_g(boundary_xt):
    return func_u_sol(boundary_xt)


def func_h(x):
    h = 2 * torch.sin(math.pi / 2 * x[:, 0]) * torch.cos(math.pi / 2 * x[:, 1])
    return h


def func_c(y_output_u):
    return -y_output_u


def func_a(xt, i, j):
    if i==j:
        return torch.ones(xt.shape[0], xt.shape[2], 1)
    else:
        return torch.zeros(xt.shape[0], xt.shape[2], 1)


def func_b(xt, i):
    return torch.zeros(xt.shape[0], xt.shape[2], 1)


def func_w(
        x):  # returns 1 for positions in the domain and 0 for those on the boundary so that our test function has support in the domain
    lens = x.shape
    w_bool = torch.gt(1 - torch.abs(x), torch.zeros(lens).to(device)) & torch.gt(torch.abs(x),
                                                                                 torch.zeros(lens).to(device))
    w_val = torch.where(w_bool, 1 - torch.abs(x) + torch.abs(x), torch.zeros(lens).to(device))
    return (w_val)  # .view(-1, 1))


''' # Data'''


class Comb_loader(Dataset):
    '''
    This class is the dataset loader. It will return the domain data for both networks as well as the border domain data.
    This data loader assumes that the boundary is a hypercube.
    '''

    def __init__(self, boundary_sample_size, domain_sample_size, t_mesh_size, dim, down=0, up=1, T0=0, T=1,
                 num_workers=1):
        '''
        boundary_sample_size (int): This is the number of points on each of the faces of the hypercube
        domain_sample_size (int): This is the number of points in the domain of the hypercube
        t_mesh_size (int): This is the number of uniformly spaced time points we look at
        dim (int): This is the number of dimensions of our problem (without time)
        down (int): This value is the value repeated in the bottom coordinate of the hypercube (the coordinate is (down, down, ...)
        up (int): This value is the value repeated in the top coordinate of the hypercube
        T0 (int): This is the first time point
        T (int): This is the last time point
        num_workers (int): This is the mumber of workers the dataloader will use
        :type up: object
        '''

        super(Comb_loader).__init__()
        self.num_workers = num_workers
        self.dim = dim
        self.sides = 2 * dim
        self.boundary_sample_size = boundary_sample_size
        self.domain_sample_size = domain_sample_size
        self.t_mesh_size = t_mesh_size
        self.down = down
        self.up = up
        self.T0 = T0
        self.T = T

        assert domain_sample_size % num_workers == 0 & self.sides * boundary_sample_size % num_workers == 0, "To make the dataloader work num_workers needs to divide the size of the domain and boundary sample on all faces"

        t = torch.linspace(T0, T, t_mesh_size).unsqueeze(1).unsqueeze(2).view(1, 1, t_mesh_size).repeat(
            domain_sample_size, 1, 1)

        X, XV = [], []

        for i in range(dim):
            x = torch.Tensor(domain_sample_size, 1, t_mesh_size).uniform_(down, up)  # .requires_grad_(True)
            X.append(x)
            XV.append(x.clone())

        X.append(t)
        XV.append(t.clone())

        ones = torch.ones(boundary_sample_size, 1)
        zeros = torch.zeros(boundary_sample_size, 1)

        border = torch.Tensor(0, dim)

        for i in range(dim):
            face_data_l = torch.Tensor(boundary_sample_size, i).uniform_(down, up)
            face_data_r = torch.Tensor(boundary_sample_size, dim - 1 - i).uniform_(down, up)
            s0 = torch.cat((face_data_l.clone(), zeros.clone(), face_data_r.clone()), 1)
            s1 = torch.cat((face_data_l.clone(), ones.clone(), face_data_r.clone()), 1)
            border = torch.cat((border, s0, s1), 0)

        border = border.unsqueeze(2).repeat(1, 1, t_mesh_size)
        tb = torch.linspace(T0, T, t_mesh_size).unsqueeze(1).unsqueeze(2).view(1, 1, t_mesh_size).repeat(
            self.sides * boundary_sample_size, 1, 1)
        border = torch.cat((border, tb), 1)

        self.interioru = X
        self.interiorv = XV
        self.border = border
        self.end_int = domain_sample_size
        self.end_bor = boundary_sample_size

    def __len__(self):
        return self.num_workers

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_int = 0
            start_bor = 0
            end_int = self.end_int
            end_bor = self.end_bor
        else:
            int_size, bor_size = int(math.ceil((self.end_int) / float(worker_info.num_workers))), int(
                math.ceil((self.end_bor) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start_int, start_bor = worker_id * int_size, worker_id * bor_size
            end_int, end_bor = min(start_int + int_size, self.end_int), min(start_bor + bor_size, self.end_bor)
        points_sep = [self.interioru[i][start_int:end_int, :] for i in range(self.dim+1)]
        for i in range(self.dim+1):
            points_sep.append(self.interiorv[i][start_int:end_int, :])
        points_sep.append(self.border[start_bor:end_bor, :, :])
        return points_sep


''' # Setting Parameters '''

# dictionary with all the configurations of meshes and the problem dimension

setup = {'dim': 2,
         'domain_sample_size': 10000,
         'boundary_sample_size': 160,
         't_mesh_size': 200
         }

# for computational efficiency the functions will be evaluated here
def funcs(points, setup):
    x = torch.Tensor(setup['domain_sample_size'], 0)
    xt = torch.Tensor(setup['domain_sample_size'], 0, setup['t_mesh_size'])

    for i in range(setup['dim']):
        x = torch.cat((x, points.interioru[i][:, :, 0]), 1)

    for i in range(setup['dim'] + 1):
        xt = torch.cat((xt, points.interioru[i]), 1)

    h = func_h(x.to(device)).to(device)
    f = func_f(xt.to(device)).to(device)
    g = func_g(points.border.to(device)).unsqueeze(2).to(device)

    # this is meant to be a d by d-dimensional array containing domain_sample_size by t_mesh_size by 1 tensors
    a = torch.Tensor(setup['dim'], setup['dim'], setup['domain_sample_size'], setup['t_mesh_size'], 1)

    for i, j in product(range(setup['dim']), repeat=2):
        a[i, j] = func_a(xt, i, j)

    # this is meant to be a d-dimensional containing domain_sample_size by t_mesh_size by 1 tensors
    b = torch.Tensor(setup['dim'], setup['domain_sample_size'], setup['t_mesh_size'], 1)

    for i in range(setup['dim']):
        b[i] = func_b(xt, i)

    return h.to(device), f.to(device), g.to(device), a.to(device), b.to(device)


"""# Defining the Model"""


# this function ensures that the layers have the right weigth initialisation
def init_weights(layer):
    if type(layer) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)


class generator(torch.nn.Module):
    '''
    This function is the generator and will be the function that will give us the weak solution. It will be referred to
    as the u function further on. The function takes in x,y,t points and returns what we estimate to be the value of the
    solution at that point. This model can intake an arbitrarily long list of these inputs but all the lists need to be
    equally long. The input shape is [L, C, T] where L is the number of points, C is the number of dimensions and T is
    the number of time points.

    config (dict): this dictionary will contain all the hyperparameters ('u_layers' and 'u_hidden_dim' for the generator)
    '''

    def __init__(self, config):
        super().__init__()
        self.num_layers = config['u_layers']
        self.hidden_dim = config['u_hidden_dim']
        self.input = torch.nn.Linear(setup['dim'] + 1, self.hidden_dim)
        self.hidden = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = torch.nn.Linear(self.hidden_dim, 1)
        self.net = torch.nn.Sequential(*[
            self.input,
            *[torch.nn.ReLU(), self.hidden] * self.num_layers,
            torch.nn.Tanh(),
            self.output

        ])

    def forward(self, X):
        inp = torch.Tensor(X[0].shape[0], 0, setup['t_mesh_size']).to(device)
        for i in range(setup['dim']+1):
            inp = torch.cat((inp, X[i]), 1)
        x = self.net(inp.view(-1, setup['dim']+1)).view(-1, setup['t_mesh_size'])
        return x

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return (self.loss)


class discriminator(torch.nn.Module):
    '''
    This function is the discriminator and will be the function that will give us the test function. It will be referred
    to as the u function further on. The function takes in x,y,t points and returns the value of the test function at
    that point. This model can intake an arbitrarily long list of these inputs but all the lists need to be equally long.
    The input shape is [L, C, T] where L is the number of points, C is the number of dimensions and T is the number of
    time points.

    config (dict): this dictionary will contain all the hyperparameters ('v_layers' and 'v_hidden_dim' for the discriminator)
    '''

    def __init__(self, config):
        super().__init__()
        self.num_layers = config['v_layers']
        self.hidden_dim = config['v_hidden_dim']
        self.input = torch.nn.Linear(setup['dim'] + 1, self.hidden_dim)
        self.hidden = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = torch.nn.Linear(self.hidden_dim, 1)
        self.net = torch.nn.Sequential(*[
            self.input,
            *[torch.nn.ReLU(), self.hidden] * self.num_layers,
            torch.nn.Tanh(),
            self.output
        ])

    def forward(self, XV):
        inp = torch.Tensor(XV[0].shape[0], 0, setup['t_mesh_size']).to(device)
        for i in range(setup['dim'] + 1):
            inp = torch.cat((inp, XV[i]), 1)
        x = self.net(inp.view(-1, setup['dim']+1)).view(-1, setup['t_mesh_size'])
        return x

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return (self.loss)


# Hyperparameters

config = {
    'alpha': 1e2 * 40 * 4,
    'u_layers': 7,
    'u_hidden_dim': 20,
    'v_layers': 9,
    'v_hidden_dim': 50,
    'n1': 10,
    'n2': 5,
    'u_rate': 0.04,
    'v_rate': 0.015,
    'u_factor': 0.8,
    'v_factor': 0.95
}

''' # Loss functions '''


class loss:
    '''
    This is the loss class; in it all the various losses are calculated and we can call the losses for the generator and
    discriminator within the class (here denoted by `u` and `v` respectively). The function `I` calculates
    $\langle A[u_\theta], v_\phi \rangle$. The function `int` represents the interior loss, `init` the intial loss (at
    time T0) and `bdry` the loss on the boundary. (Note that in this non-linear example the function c is defined not as
    a Tensor but a function as it depends on the networks guesses of the function $u_\theta$.

    border (Tensor): Tensor containing all the points in our boundary with shape [L, C, T]
    alpha (int): the $\alpha=\gamma$ parameter that assigns relative weights to the initial and boundary loss
    a (Tensor): Tensor with the values of $a$ (from the general form) evaluated at the points of the domain
    b (Tensor): Tensor with the values of $b$ (from the general form) evaluated at the points of the domain
    h (Tensor): Tensor with the values of the function $h$ (from the general form) evaluated at the points of the domain
    f (Tensor): Tensor with the values of the function $f$ (from the general form) evaluated at the points of the domain
    g (Tensor): Tensor with the values of the function $g$ (from the general form) evaluated at the points of the domain
    setup (dict): dictionary with all the configurations of meshes and the problem dimension
    '''
    def __init__(self, border, alpha, a, b, h, f, g, setup, c=func_c, T=1, T0=0):
        super(loss).__init__()
        self.boundary = border
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

    def I(self, y_output_u, y_output_v, ind, X, XV):
        y_output_u.retain_grad()
        y_output_v.retain_grad()
        N = y_output_u.shape[0]
        V = 1       # Volume of \Omega
        # This section was strangely computationally intensive and I have commented out for the time being. All it did
        # was to turn points of the test function on the boundary to be 0. (In the paper they assume $supp\phi \subset
        # \Omega$)
        '''
        w = torch.ones(y_output_v.shape)
        for i in range(setup['dim']):
            f = func_w(XV[i])
            w = torch.mul(w, f)
        '''
        phi = y_output_v  # torch.mul(y_output_v, w)
        y_output_u.backward(torch.ones_like(y_output_u).to(device), retain_graph=True)
        du = {}
        for i in range(self.setup['dim']):
            du['du_' + str(i)] = X[i].grad.view(-1, setup['t_mesh_size'], 1)
        phi.backward(torch.ones_like(phi).to(device), retain_graph=True)
        dphi = {}
        for i in range(self.setup['dim'] + 1):
            dphi['dphi_' + str(i)] = XV[i].grad.view(-1, setup['t_mesh_size'], 1)
        s1 = V * (y_output_u[:, -1] * phi[:, -1] - self.h[ind * N:(ind + 1) * N] * phi[:, 0]) / N
        s2 = (self.T - self.T0) * V * (y_output_u * dphi['dphi_2'].squeeze(2)) / self.setup[
            't_mesh_size'] / N
        s31 = 0
        for i, j in product(range(self.setup['dim']), repeat=2):
            s31 += self.a[i, j, ind * N:(ind + 1) * N, :, :] * dphi['dphi_' + str(i)] * du['du_' + str(j)]
        s32 = 0
        for i in range(self.setup['dim']):
            s32 += self.b[i, ind * N:(ind + 1) * N, :, :] * phi.unsqueeze(2) * du['du_' + str(i)]
        s3 = (self.T - self.T0) * V * (s31.squeeze(2) + s32.squeeze(2) + self.c(y_output_u) * y_output_u * phi - self.f[ind * N:(ind + 1) * N,:] * phi) / \
             self.setup['t_mesh_size'] / N
        I = torch.sum(s1 - torch.sum(s2 - s3, 1), 0)
        for i in X:
            i.grad.data.zero_()
        for i in XV:
            i.grad.data.zero_()
        return I

    def init(self, y_output_u, ind):
        N = y_output_u.shape[0]
        return torch.mean((y_output_u[:, 0] - self.h[ind * N:(ind + 1) * N]) ** 2)

    # initially had all variables feed with grad
    def bdry(self, ind, u_net, N):
        return torch.mean((u_net([self.boundary[ind * N:(ind + 1) * N, i, :].unsqueeze(1) for i in range(setup['dim']+1)]) - self.g[ind * N:(ind + 1) * N].squeeze(
            2)) ** 2)

    def int(self, y_output_u, y_output_v, ind, X, XV):
        # x needs to be the set of points set plugged into net_u and net_v
        return torch.log(self.I(y_output_u, y_output_v, ind, X, XV) ** 2) - torch.log(torch.sum(y_output_v ** 2))

    def u(self, y_output_u, y_output_v, ind, u_net, X, XV):
        N = y_output_u.shape[0]
        return self.int(y_output_u, y_output_v, ind, X, XV) + self.alpha * (
                    self.init(y_output_u, ind) + self.bdry(ind, u_net, N))

    def v(self, y_output_u, y_output_v, ind, X, XV):
        return -self.int(y_output_u, y_output_v, ind, X, XV)

def L_norm(X, predu, p):
    # p is the p in L^p
    xt = torch.Tensor(X[0].shape[0], 0, X[0].shape[2])
    for i in X:
        xt = torch.cat((xt, i), 1)
    u_sol = func_u_sol(xt).to(device)
    return (torch.mean(torch.pow(torch.abs(u_sol-predu), p)))**(1/p)

# TODO: create a projection function
''' # Training '''


def train(config, setup, iterations):
    points = Comb_loader(setup['boundary_sample_size'], setup['domain_sample_size'], setup['t_mesh_size'], setup['dim'])
    ds = DataLoader(points, num_workers=points.num_workers)

    h, f, g, a, b = funcs(points, setup)

    n1 = config['n1']
    n2 = config['n2']

    # neural network models
    u_net = torch.nn.DataParallel(generator(config)).to(device)
    v_net = torch.nn.DataParallel(discriminator(config)).to(device)

    u_net.apply(init_weights)
    v_net.apply(init_weights)

    # TODO: test that the ind allows for correct X
    Loss = loss(points.border, config['alpha'], a, b, h, f, g, setup)

    for k in range(iterations):

        # optimizers for WAN
        optimizer_u = torch.optim.Adam(u_net.parameters(), lr=config['u_rate'])
        optimizer_v = torch.optim.Adam(v_net.parameters(), lr=config['v_rate'])

        scheduler_u = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_u, factor=config['u_factor'], patience=100)
        scheduler_v = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_v, factor=config['v_factor'], patience=100)

        for i in range(n1):
            for ind, data in enumerate(ds):
                for i in range(2*setup['dim']+2):
                    data[i] = data[i].squeeze(0).to(device).requires_grad_(True)
                X = data[:setup['dim']+1]
                XV = data[setup['dim']+1: 2*setup['dim']+2]
                prediction_v = v_net(XV)
                prediction_u = u_net(X)
                loss_u = Loss.u(prediction_u, prediction_v, ind, u_net, X, XV)
                # writer.add_scalar("Loss", loss_u, k)
                optimizer_u.zero_grad()
                loss_u.backward(retain_graph=True)
                optimizer_u.step()
            scheduler_u.step(loss_u)

        for j in range(n2):
            for ind, data in enumerate(ds):
                for i in range(2 * setup['dim'] + 2):
                    data[i] = data[i].squeeze(0).to(device).requires_grad_(True)
                X = data[:setup['dim'] + 1]
                XV = data[setup['dim'] + 1: 2 * setup['dim'] + 2]
                prediction_v = v_net(XV)
                prediction_u = u_net(X)
                loss_v = Loss.v(prediction_u, prediction_v, ind, X, XV)
                optimizer_v.zero_grad()
                loss_v.backward(retain_graph=True)
                optimizer_v.step()
            scheduler_v.step(loss_v)

        if k % 1 == 0:
            lu, lv = loss_u.item(), loss_v.item()
            print('iteration: ' + str(k), 'Loss u: ' + str(lu), 'Loss v: ' + str(lv))
            X = points.interioru
            predu = u_net(X)
            print('L^1 norm ' + str(L_norm(X, predu, 1).item()))

'''
# The code below allows us to evaluate what the loss function will give when u is the true solution
points = Comb_loader(setup['boundary_sample_size'], setup['domain_sample_size'], setup['t_mesh_size'], setup['dim'])
h, f, g, a, b = funcs(points, setup)
Loss = loss(points.border, config['alpha'], a, b, h, f, g, setup)

for i in range(setup['dim']+1):
    points.interioru[i] = points.interioru[i].requires_grad_(True)
    points.interiorv[i] = points.interiorv[i].requires_grad_(True)

xt = torch.Tensor(setup['domain_sample_size'], 0, setup['t_mesh_size']).requires_grad_(True)
for i in points.interioru:
    xt = torch.cat((xt.requires_grad_(True), i), 1)

xt.retain_grad()
points.interioru[0].retain_grad()

v_net = discriminator(config)

print(Loss.I(func_u_sol(xt), v_net(points.interiorv), 0, points.interioru, points.interiorv))
print(Loss.int(func_u_sol(xt), v_net(points.interiorv), 0, points.interioru, points.interiorv))
# initial loss and boundary loss obviously work
#'''

train(config, setup, 100)


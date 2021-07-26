from data.dataset import Path, FlexibleODEDataset, SubsampleDataset
from data.scalers import TrickScaler
from data.intervals import FixedIntervalSampler, RandomSampler, BatchIntervalSampler, create_interval_dataloader
from loss import loss
from data.functions import torch_ffill
from rdeint import rdeint
from model import NeuralRDE
import signatory
import itertools
import math
import torch
from torch.utils.data import DataLoader, Dataset
from itertools import product
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import json


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
print(device)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.init()

'''
# Setting the specific problem to solve
'''

# We denote spatial coordinates with time as 'xt' and 'x' without


def func_u_sol(X):
    u = 2 * torch.exp(-X[:, :, 0]) * 10 ** (3*setup['dim']) * np.prod([torch.sin(math.pi / 2 * X[:, :, i+1] + (math.pi * (i))/2) for i in range(setup['dim'])])
    return(u)


def func_f(X):
    f = (setup['dim'] * math.pi ** 2 / 4 - 1) * func_u_sol(X) - func_u_sol(X) ** 2
    return f


def func_g(BX):
    return func_u_sol(BX)


def func_h(X):
    h = 2 * np.prod([torch.sin(math.pi / 2 * X[:, i+1] + (math.pi * (i))/2) for i in range(setup['dim'])])
    return h


def func_c(y_output_u):
    return -y_output_u


# TODO: go over these functions
def func_a(xt, i, j):
    if i == j:
        return torch.ones(xt.shape[:-1])
    else:
        return torch.zeros(xt.shape[:-1])


def func_b(xt, i):
    return torch.zeros(xt.shape[:-1])


''' # Data'''

# TODO: move combloader to dataset


class Comb_loader(Dataset):
    '''
    This class is the dataset loader. It will return the domain data for both networks as well as the border domain data.
    This data loader assumes that the boundary is a hypercube.
    '''

    def __init__(self, intervals, step, batch_size, depth, border_batch_size, dim, down=-1, up=1, T0=0, T=1,
                 num_workers=1):
        # TODO: edit the comments to match
        '''
        boundary_sample_size (int): This is the number of points on each of the faces of the hypercube
        domain_sample_size (int): This is the number of points in the domain of the hypercube
        dim (int): This is the number of dimensions of our problem (without time)
        down (int): This value is the value repeated in the bottom coordinate of the hypercube (the coordinate is (down, down, ...)
        up (int): This value is the value repeated in the top coordinate of the hypercube
        T0 (int): This is the first time point
        T (int): This is the last time point
        num_workers (int): This is the mumber of workers the dataloader will use
        '''

        super(Comb_loader).__init__()
        self.num_workers = num_workers
        self.dim = dim
        self.sides = 2 * dim
        self.intervals = intervals
        self.step = step
        self.batch_size = batch_size
        self.depth = depth
        self.t_mesh_size = int(step*intervals)
        self.npaths = batch_size
        self.border_batch_size = border_batch_size
        self.bnpaths = int(border_batch_size / self.sides)
        self.down = down
        self.up = up
        self.T0 = T0
        self.T = T

        # TODO: change this state so that different ratios can exist

        assert self.npaths == (self.sides * self.bnpaths), "Need equality to ensure even samples in all subsamples from the dataloader"

        # TODO: check gradients work (the idea is that I can call the grad of X[i] after I have moved txy to cuda)
        t, i = torch.sort(torch.Tensor(1, self.t_mesh_size, 1).uniform_(T0, T), 1)
        t = t.repeat(self.npaths, 1, 1)
        t[:, 0, :] = T0 * torch.ones_like(t[:, 0, :])
        t[:, -1, :] = T * torch.ones_like(t[:, 0, :])

        X = [t.requires_grad_(True)]
        [X.append(torch.Tensor(self.npaths, 1, 1).uniform_(down, up).repeat(1, self.t_mesh_size, 1).requires_grad_(True)) for i in range(dim)]
        XV = [i.detach().clone().requires_grad_(True) for i in X]

        ups = up * torch.ones(self.bnpaths, self.t_mesh_size, 1)
        downs = down * torch.ones(self.bnpaths, self.t_mesh_size, 1)

        sd = [torch.cat((torch.Tensor(self.bnpaths, 1, i).uniform_(down, up).repeat(1, self.t_mesh_size, 1), downs.clone(), torch.Tensor(self.bnpaths, 1, dim - 1 - i).uniform_(down, up).repeat(1, self.t_mesh_size, 1)), 2) for i in range(dim)]
        su = [torch.cat((torch.Tensor(self.bnpaths, 1, i).uniform_(down, up).repeat(1, self.t_mesh_size, 1), ups.clone(), torch.Tensor(self.bnpaths, 1, dim - 1 - i).uniform_(down, up).repeat(1, self.t_mesh_size, 1)), 2) for i in range(dim)]

        btxy = torch.cat(tuple(sd + su), 0)

        idx = torch.randperm(int(self.bnpaths * self.sides))
        btxy = btxy[idx, :, :]

        tb, i = torch.sort(torch.Tensor(1, self.t_mesh_size, 1).uniform_(T0, T), 1)
        tb = tb.repeat(int(self.bnpaths * self.sides), 1, 1)
        tb[:, 0, :] = self.T0 * torch.ones_like(tb[:, 0, :])
        tb[:, -1, :] = self.T * torch.ones_like(tb[:, -1, :])
        self.btxy = torch.cat((tb, btxy), 2)

        border = [btxy[:, :, i] for i in range(btxy.shape[2])]

        self.interioru = X
        self.interiorv = XV
        self.border = border

        self.txy = torch.cat(tuple(X), 2)

        self.init = self.txy[:, 0, 1:]

        self.sampler = FixedIntervalSampler(self.t_mesh_size, self.step, from_start=True, include_end=False)

        self.dataset = FlexibleODEDataset(self.txy, self.btxy, depth=depth)

        self.input_dim = self.dataset.input_dim

        self.dataloader = create_interval_dataloader(self.dataset, self.sampler, int(batch_size))

''' # Setting Parameters '''

# dictionary with all the configurations of meshes and the problem dimension

setup = {'dim': 5,
         'intervals': 5,
         'batch_size': 80*5,
         'depth': 1,
         'border_batch_size': 80*5,
         'step': 4
         }

# for computational efficiency the functions will be evaluated here
def funcs(points, setup):
    with torch.no_grad():
        xt = points.txy.to(device)

        h = func_h(xt[:, 0, :])
        f = func_f(xt)
        g = func_g(points.btxy.to(device))

        # this is meant to be a d by d-dimensional array containing domain_sample_size by 1 tensors
        a = torch.Tensor(setup['dim'], setup['dim'], setup['batch_size'], setup['intervals']*setup['step'])

        for i, j in product(range(setup['dim']), repeat=2):
            a[i, j] = func_a(xt, i, j)

        # this is meant to be a d-dimensional containing domain_sample_size by 1 tensors
        b = torch.Tensor(setup['dim'], setup['batch_size'], setup['intervals']*setup['step'])

        for i in range(setup['dim']):
            b[i] = func_b(xt, i)

    return h, f, g, a.to(device), b.to(device)


"""# Defining the Model"""


# this function ensures that the layers have the right weigth initialisation
def init_weights(layer):
    if type(layer) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)

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
        self.net.cuda()

    def forward(self, XV):
        inp = torch.cat(tuple(XV), 2)
        x = self.net(inp)
        return x

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return (self.loss)


# Hyperparameters

config = {
    'alpha': 1e4 * 400 * 25,  #tune.loguniform(1e4, 1e7),   #
    'u_layers': 8,
    'u_hidden_dim': 20,
    'u_hidden_hidden_dim': 10,
    'v_layers': 9,
    'v_hidden_dim': 50,
    'n1': 2,
    'n2': 1,
    'u_rate': 0.015,  #tune.loguniform(0.01, 0.2),
    'v_rate': 0.04,  #tune.loguniform(0.01, 0.2),
    'u_factor': 1-1e-10
}


''' # Auxillary Funcs '''

def L_norm(X, predu, p, T0=0, T=1):
    # p is the p in L^p
    xt = torch.cat(tuple(X), 2)
    u_sol = func_u_sol(xt).to(device).unsqueeze(2)
    return ((T-T0) * 2 ** setup['dim'] * torch.mean(torch.pow(torch.abs(u_sol - predu), p))) ** (1 / p)

def rel_err(X, predu):
    xt = torch.cat(tuple(X), 2)
    u_sol = func_u_sol(xt).to(device).unsqueeze(2)
    rel = torch.abs(torch.div(torch.mean(u_sol - predu), torch.mean(u_sol)))
    return rel

#'''
def proj(u_net, axes=[0, 1], down=-1, up=1, T=1, T0=0, save=False, show=True, resolution=100, colours=8, iteration=0):
    # Assumes hypercube
    assert len(axes) == 2, 'There can only be two axes in the graph to be able to display them'
    assert axes[0] == 0, 'For the ODEINT case we can only currently plot time against a spatial coordinate'

    xt = torch.Tensor(resolution, resolution, setup['dim'] + 1).to(device)

    if 0 in axes:
        t_mesh = torch.linspace(T0, T, resolution)
    else:
        t_mesh = torch.linspace(down, up, resolution)
        xt[:, -1], i = torch.sort(torch.Tensor(resolution).uniform_(T0, T), 0)

    x_mesh = torch.linspace(down, up, resolution)
    mesh1, mesh2 = torch.meshgrid(x_mesh, t_mesh)
    xt[:, :, 0] = mesh2
    xt[:, :, axes[1]] = mesh1

    for i in list(set(range(setup['dim'] + 1)) - set(axes)):
        xt[:, :, i] = (up+down)/4 * torch.ones(resolution, resolution)

    u_sol = func_u_sol(xt).to(device)
    predu = u_net((xt[:,0,:], xt)).to(device).detach()

    error = predu - u_sol.unsqueeze(2)

    plt.clf()
    fig, ax = plt.subplots(3)
    aset = ax[0].contourf(x_mesh.numpy(), t_mesh.numpy(), u_sol.view(resolution, resolution).cpu().numpy(), colours)
    bset = ax[1].contourf(x_mesh.numpy(), t_mesh.numpy(), predu.view(resolution, resolution).cpu().numpy(), colours)
    cset = ax[2].contourf(x_mesh.numpy(), t_mesh.numpy(), error.view(resolution, resolution).cpu().numpy(), colours)
    fig.colorbar(aset, ax=ax[0])
    fig.colorbar(bset, ax=ax[1])
    fig.colorbar(cset, ax=ax[2])
    ax[0].set_title('Correct Solution, Guess and Error')

    if save:
        plt.savefig('NODE_plots_time_' + str(iteration) + '.png')
        print('Saved')

    if show:
        plt.show()
        print('Displayed')


#'''

''' # Training '''

params = {**config, **setup, **{'iterations': int(200)}}

def train(params, checkpoint_dir=None):
    i = iter(params.items())
    config = dict(itertools.islice(i, 11))
    setup = dict(itertools.islice(i, 6))
    iterations = dict(i)['iterations']

    n1 = config['n1']
    n2 = config['n2']

    points = Comb_loader(setup['intervals'], setup['step'], setup['batch_size'], setup['depth'],
                         setup['border_batch_size'], setup['dim'])

    # neural network models
    u_net = torch.nn.DataParallel(NeuralRDE(points.input_dim, config["u_hidden_dim"], 1, func_h, hidden_hidden_dim=config['u_hidden_hidden_dim'], num_layers=config['u_layers'], return_sequences=True).to(device)).to(device)
    v_net = torch.nn.DataParallel(discriminator(config)).to(device)

    u_net.apply(init_weights)
    v_net.apply(init_weights)

    # optimizers for WAN
    optimizer_u = torch.optim.Adam(u_net.parameters(), lr=config['u_rate'])
    optimizer_v = torch.optim.Adam(v_net.parameters(), lr=config['v_rate'])

    scheduler_u = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_u, factor=config['u_factor'], patience=15)

    losses = []

    for k in range(iterations):

        points = Comb_loader(setup['intervals'], setup['step'], setup['batch_size'], setup['depth'],
                             setup['border_batch_size'], setup['dim'])
        dl = points.dataloader

        h, f, g, a, b = funcs(points, setup)
        Loss = loss(config['alpha'], a, b, h, f, g, setup, func_c)

        for i in range(n1):
             for ind, (data, bdata) in enumerate(dl):
                optimizer_u.zero_grad()
                N = data[0].shape[0]
                prediction_v = v_net([l[ind*N:(ind+1)*N] for l in points.interiorv])
                prediction_u = u_net(data)
                loss_u = Loss.u(prediction_u, prediction_v, ind, u_net, points.interioru, points.interiorv, bdata)
                losses.append(loss_u.item())
                with open("NODE_lossu.txt", "w") as fp:
                    json.dump(losses, fp)
                # TODO: check that the loss actually works
                loss_u.backward(retain_graph=True)
                optimizer_u.step()

        # TODO: implement a timestep dependent scheduler
        scheduler_u.step(np.mean(losses[-10:]))
        #print('mean of losses: ' + str(np.mean(losses[-10:])))
        #print(optimizer_u.param_groups[0]['lr'])

        for j in range(n2):
            for ind, (data, bdata) in enumerate(dl):
                optimizer_v.zero_grad()
                N = data[0].shape[0]
                prediction_v = v_net([l[ind * N:(ind + 1) * N] for l in points.interiorv])
                prediction_u = u_net(data)
                loss_v = Loss.v(prediction_u, prediction_v, ind, points.interioru, points.interiorv)
                loss_v.backward(retain_graph=True)
                optimizer_v.step()

        if k % 5 == 0:
            lu, lv = loss_u.item(), loss_v.item()
            print('iteration: ' + str(k), 'Loss u: ' + str(lu), 'Loss v: ' + str(lv))
            L1 = L_norm(points.interioru, prediction_u, 1).item()
            #tune.report(L1=L1)
            print('L^1 norm ' + str(L1))
            #print('Relative Error ' + str.format('{0:.3f}', rel_err(points.interioru, prediction_u)) + '%')
            print('Learning rate ' + str(optimizer_u.param_groups[0]['lr']))
            if k % 5 == 0:
                proj(u_net, axes=[0, 1], resolution=200, colours=20, iteration=k, save=True, show=False)




'''
# The code below allows us to evaluate what the loss function will give when u is the true solution
points = Comb_loader(setup['boundary_sample_size'], setup['domain_sample_size'], setup['dim'])
h, f, g, a, b = funcs(points, setup)
Loss = loss(points.border, config['alpha'], a, b, h, f, g, setup, points.init.to(device))

interioru = [i.to(device).requires_grad_(True) for i in points.interioru]
interiorv = [i.to(device).requires_grad_(True) for i in points.interiorv]

xt = torch.Tensor(setup['domain_sample_size'], 0).to(device).requires_grad_(True)
for i in interioru:
    xt = torch.cat((xt, i), 1)

xt.retain_grad()

v_net = discriminator(config)

print(Loss.I(func_u_sol(xt).unsqueeze(1), v_net(interiorv), 0, interioru, interiorv, func_u_sol, v_net))

print(Loss.int(func_u_sol(xt).unsqueeze(1), v_net(interiorv), 0, interioru, interiorv, func_u_sol, v_net))
# initial loss and boundary loss obviously work
#'''

train(params)

'''
analysis = tune.run(
    train,
    num_samples=100,
    scheduler=ASHAScheduler(metric="L1", mode="min", grace_period=200, max_t=2e4),
    config=params,
    verbose=3,
    resources_per_trial={"cpu": 1, "gpu": 0.4},
    keep_checkpoints_num=1,
    checkpoint_score_attr="min-L1",
    resume=False
)


best_trial = analysis.get_best_trial(metric="L1", mode="min", scope='all')
print(best_trial)
#'''
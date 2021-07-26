from timeit import timeit
import time
from data.dataset import Path, FlexibleODEDataset, SubsampleDataset
from data.scalers import TrickScaler
from data.intervals import FixedIntervalSampler, RandomSampler, BatchIntervalSampler, create_interval_dataloader
from loss import loss
from data.functions import torch_ffill
from rdeint import rdeint
from model import NeuralRDE
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
    u = 2 * torch.sin(math.pi / 2 * X[:, :, 1]) * torch.cos(math.pi / 2 * X[:, :, 2]) * torch.exp(-X[:, :, 0])
    return(u)


def func_f(X):
    f = (math.pi ** 2 - 2) * torch.sin(math.pi / 2 * X[:, :, 1]) * torch.cos(math.pi / 2 * X[:, :, 2]) * torch.exp(-X[:, :, 0]) - 4 * torch.sin(math.pi / 2 * X[:, :, 1]) ** 2 * torch.cos(math.pi / 2 * X[:, :, 2]) ** 2 * torch.exp(-2*X[:, :, 0])
    return f


def func_g(BX):
    return func_u_sol(BX)


def func_h(X):
    h = 2 * torch.sin(math.pi / 2 * X[:, 1]) * torch.cos(math.pi / 2 * X[:, 2])
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


class NSphere_Tcone:
    def __init__(self, r, dim, T0, T, N_t):
        self.r = r
        self.dim = dim
        self.T0 = T0
        self.T = T
        self.N_t = N_t
        self.times, i = torch.sort(torch.Tensor(self.N_t).uniform_(T0, T), 0)
        self.times[0], self.times[-1] = T0, T

    def surf(self, N):
        normal_deviates = np.random.normal(size=(self.dim, N))

        radius = np.sqrt((normal_deviates ** 2).sum(axis=0))
        return normal_deviates / radius

    def interior(self, N_r):
        points = self.surf(N_r)
        points *= np.random.rand(N_r)**(1/self.dim)
        time_data = self.times.repeat(N_r, 1).unsqueeze(2)

        datapoints = []
        k = self.N_t

        for t in self.times.numpy()[::-1]:
            idx = np.sqrt(np.sum(points**2, 0)) < (1-t)
            npoints = torch.transpose(torch.from_numpy(points[:, idx]), 0, 1).unsqueeze(1).repeat(1, k, 1)
            points = np.delete(points, idx, 1)
            if npoints.shape[0] != 0:
                datapoints.append(torch.cat((time_data[:npoints.shape[0], :k], npoints), 2).requires_grad_(True))
            k -= 1

        return datapoints[::-1]

    def boundary(self, N_b):
        datapoints = []

        for t in self.times.numpy():
            n = int(N_b * (1-t)**self.dim)
            points = self.surf(n) * (1-t)
            points = torch.transpose(torch.from_numpy(points), 0, 1).unsqueeze(1)
            ones = torch.ones(n, 1, 1)
            if n != 0:
                datapoints.append(torch.cat((t*ones, points), 2).requires_grad_(True))

        return datapoints


class Comb_loader(Dataset):

    def __init__(self, N_r, N_b, shape):
        self.N_r = N_r
        self.N_b = N_b
        self.shape = shape
        self.interioru = self.shape.interior(self.N_r)
        self.interiorv = self.shape.interior(self.N_r)
        self.boundary = self.shape.boundary(self.N_b)

        m = min(len(self.interioru), len(self.interiorv))

        self.interior_u = [self.interioru[i] for i in range(m) if self.interiorv[i].shape[1] == self.interioru[i].shape[1]]
        self.interior_v = [self.interiorv[i] for i in range(m) if self.interiorv[i].shape[1] == self.interioru[i].shape[1]]

    def __len__(self):
        return max(len(self.interior), len(self.boundary))

    def __getitem__(self, idx):
        data_u = self.interior_u[idx].to(device)
        data_v = self.interior_v[idx].to(device)
        m = min(len(data_v), len(data_u))
        r = (data_u[:m], data_v[:m], self.boundary[idx].to(device))
        return r

#k = Comb_loader(20, 10, NSphere_Tcone(1, 3, 0, 1, 5))[2]

''' # Setting Parameters '''

# dictionary with all the configurations of meshes and the problem dimension

setup = {'dim': 5,
         'intervals': 5,
         'batch_size': 800*5,
         'depth': 1,
         'border_batch_size': 800*5,
         'step': 4
         }

# for computational efficiency the functions will be evaluated here
def funcs(data, bdata, setup):
    with torch.no_grad():
        xt = data.clone().detach().to(device)

        h = func_h(xt[:, 0, :])
        f = func_f(xt)
        g = func_g(bdata.to(device))

        # this is meant to be a d by d-dimensional array containing domain_sample_size by 1 tensors
        a = torch.Tensor(setup['dim'], setup['dim'], data.shape[0], data.shape[1])

        for i, j in product(range(setup['dim']), repeat=2):
            a[i, j] = func_a(xt, i, j)

        # this is meant to be a d-dimensional containing domain_sample_size by 1 tensors
        b = torch.Tensor(setup['dim'], data.shape[0], data.shape[1])

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
        self.net.cuda().double()

    def forward(self, XV):
        x = self.net(XV.double())
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
    u_sol = func_u_sol(X).to(device).unsqueeze(2)
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
        xt[:, :, i] = (up+down)/2 * torch.ones(resolution, resolution)

    u_sol = func_u_sol(xt).to(device)
    predu = u_net((xt[:,0,:], xt)).to(device).detach().squeeze()

    error = predu - u_sol

    plt.clf()
    fig, ax = plt.subplots(3)
    aset = ax[0].contourf(t_mesh.numpy(), x_mesh.numpy(), u_sol.cpu().numpy(), colours)
    bset = ax[1].contourf(t_mesh.numpy(), x_mesh.numpy(), predu.cpu().numpy(), colours)
    cset = ax[2].contourf(t_mesh.numpy(), x_mesh.numpy(), error.cpu().numpy(), colours)
    fig.colorbar(aset, ax=ax[0])
    fig.colorbar(bset, ax=ax[1])
    fig.colorbar(cset, ax=ax[2])
    ax[0].set_title('Correct Solution, Guess and Error')

    if save:
        plt.savefig('NODE_plots_time_' + str(iteration) + '_lr_' + str(config['u_rate']) + '.png')
        print('Saved')

    if show:
        plt.show()
        print('Displayed')


#'''

''' # Training '''

params = {**config, **setup, **{'iterations': int(301)}}

def train(params, checkpoint_dir=None):
    i = iter(params.items())
    config = dict(itertools.islice(i, 11))
    setup = dict(itertools.islice(i, 6))
    iterations = dict(i)['iterations']

    n1 = config['n1']
    n2 = config['n2']

    points = Comb_loader(setup['batch_size'], setup['border_batch_size'], NSphere_Tcone(1, setup['dim'], 0, 1, setup['intervals']*setup['step']))

    # neural network models
    u_net = NeuralRDE(setup['dim'], config["u_hidden_dim"], 1, func_h, setup, hidden_hidden_dim=config['u_hidden_hidden_dim'], num_layers=config['u_layers'], return_sequences=True).to(device)   #torch.nn.DataParallel().to(device)
    v_net = discriminator(config).to(device) #torch.nn.DataParallel()

    u_net.apply(init_weights)
    v_net.apply(init_weights)

    # optimizers for WAN
    optimizer_u = torch.optim.Adam(u_net.parameters(), lr=config['u_rate'])
    optimizer_v = torch.optim.Adam(v_net.parameters(), lr=config['v_rate'])

    scheduler_u = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_u, factor=config['u_factor'], patience=15)

    losses = []
    l1s = []
    times = []

    for k in range(iterations):

        points = Comb_loader(setup['batch_size'], setup['border_batch_size'], NSphere_Tcone(1, setup['dim'], 0, 1, setup['intervals']*setup['step']))

        for i in range(n1):
             for (datau, datav, bdata) in points:
                h, f, g, a, b = funcs(datau.clone().detach(), bdata, setup)
                Loss = loss(config['alpha'], a, b, h, f, g, setup, func_c)
                optimizer_u.zero_grad()
                prediction_v = v_net(datav.to(device))
                prediction_u = u_net(datau)
                loss_u = Loss.u(prediction_u, prediction_v, u_net, datau, datav, bdata, func_g)
                losses.append(loss_u.item())
                with open("NLoss_lossu_hiddim_"+str(config['u_hidden_dim'])+ "_depth_"+ str(config['u_layers']) +".txt", "w") as fp:
                    json.dump(losses, fp)
                l1s.append(L_norm(datau.clone().detach(), prediction_u, 1).item())
                with open("NLoss_L1_hiddim_"+str(config['u_hidden_dim'])+ "_depth_"+ str(config['u_layers']) +".txt", "w") as fp:
                    json.dump(l1s, fp)
                times.append(time.time())
                with open("NLoss_time_hiddim_"+str(config['u_hidden_dim'])+ "_depth_"+ str(config['u_layers']) + ".txt", "w") as fp:
                    json.dump(times, fp)

                # TODO: check that the loss actually works
                loss_u.backward(retain_graph=True)
                optimizer_u.step()

        for j in range(n2):
            for (datau, datav, bdata) in points:
                h, f, g, a, b = funcs(datau.clone().detach(), bdata, setup)
                Loss = loss(config['alpha'], a, b, h, f, g, setup, func_c)
                optimizer_v.zero_grad()
                prediction_v = v_net(datav.to(device))
                prediction_u = u_net(datau)
                loss_v = Loss.v(prediction_u, prediction_v, datau, datav)
                loss_v.backward(retain_graph=True)
                optimizer_v.step()

        if k % 100 == 0:
            lu, lv = loss_u.item(), loss_v.item()
            print('iteration: ' + str(k), 'Loss u: ' + str(lu), 'Loss v: ' + str(lv))
            (datau, datav, bdata) = next(iter(points))
            prediction_u = u_net(datau)
            L1 = L_norm(datau, prediction_u, 1).item()
            #tune.report(L1=L1)
            print('L^1 norm ' + str(L1))
            #print('Relative Error ' + str.format('{0:.3f}', rel_err(points.interioru, prediction_u)) + '%')
            print('Learning rate ' + str(optimizer_u.param_groups[0]['lr']))
            #if k % 100 == 0:
                #proj(u_net, axes=[0, 1], resolution=200, colours=20, iteration=k, save=False, show=True)




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

#print(timeit('train(params)', 'from __main__ import train, params', number=100))

'''
lsl = np.arange(4, 10, 1)
lsd = np.arange(10, 30, 4)

for j in lsd:
    params['u_hidden_dim'] = int(j)
    for i in lsl:
        params['u_layers'] = int(i)
        train(params)
'''

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
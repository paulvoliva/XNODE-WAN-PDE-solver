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

def func_u_sol(xt):
    l = xt.shape[0]
    u = 2 * torch.sin(math.pi / 2 * xt[:, 0]) * torch.cos(math.pi / 2 * xt[:, 1]) * torch.exp(-xt[:, 2])
    return (u)


def func_f(xt):
    l = xt.shape[0]
    f = (math.pi ** 2 - 2) * torch.sin(math.pi / 2 * xt[:, 0]) * torch.cos(math.pi / 2 * xt[:, 1]) * torch.exp(
        -xt[:, 2]) - 4 * torch.sin(math.pi / 2 * xt[:, 0]) ** 2 * torch.cos(
        math.pi / 2 * xt[:, 1]) ** 2 * torch.exp(-2 * xt[:, 2])
    return (f)


def func_g(boundary_xt):
    return func_u_sol(boundary_xt)


def func_h(x):
    h = 2 * torch.sin(math.pi / 2 * x[:, 0]) * torch.cos(math.pi / 2 * x[:, 1])
    return h


def func_c(y_output_u):
    return -y_output_u


def func_a(xt, i, j):
    if i == j:
        return torch.ones(xt.shape[0], 1)
    else:
        return torch.zeros(xt.shape[0], 1)


def func_b(xt, i):
    return torch.zeros(xt.shape[0], 1)


def func_w(x):  # returns 1 for positions in the domain and 0 for those on the boundary so that our test function has support in the domain
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

    def __init__(self, boundary_sample_size, domain_sample_size, dim, down=-1, up=1, T0=0, T=1,
                 num_workers=1):
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
        assert boundary_sample_size % self.sides == 0, 'For each of the sides we need the same amount of points so `boundary_sample_size` must be a multiple of `2*dim`'
        self.boundary_sample_size = boundary_sample_size
        self.domain_sample_size = domain_sample_size
        self.down = down
        self.up = up
        self.T0 = T0
        self.T = T

        assert domain_sample_size % num_workers == 0 & self.sides * boundary_sample_size % num_workers == 0, "To make the dataloader work num_workers needs to divide the size of the domain and boundary sample on all faces"

        t = torch.Tensor(domain_sample_size, 1).uniform_(T0, T)

        X = [torch.Tensor(domain_sample_size, 1).uniform_(down, up) for i in range(dim)]
        X.append(t)
        XV = [torch.Tensor(domain_sample_size, 1).uniform_(down, up) for i in range(dim)]
        XV.append(t.clone())

        side_size = int(boundary_sample_size / self.sides)

        ups = up * torch.ones(side_size, 1)
        downs = down * torch.ones(side_size, 1)

        border = torch.Tensor(0, dim)

        sd = [torch.cat((torch.Tensor(side_size, i).uniform_(down, up), downs.clone(), torch.Tensor(side_size, dim - 1 - i).uniform_(down, up)), 1) for i in range(dim)]
        su = [torch.cat((torch.Tensor(side_size, i).uniform_(down, up), ups.clone(), torch.Tensor(side_size, dim - 1 - i).uniform_(down, up)), 1) for i in range(dim)]

        for i, j in zip(sd, su):
            border = torch.cat((border, i, j), 0)

        tb = torch.Tensor(boundary_sample_size, 1).uniform_(T0, T)
        border = torch.cat((border, tb), 1)

        init = torch.Tensor(boundary_sample_size, dim).uniform_(down, up)

        self.interioru = X
        self.interiorv = XV
        self.border = border
        self.init = init
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
        points_sep = [self.interioru[i][start_int:end_int, :] for i in range(self.dim + 1)] + [self.interiorv[i][start_int:end_int, :] for i in range(self.dim + 1)]
        points_sep.append(self.border[start_bor:end_bor, :])
        return points_sep


''' # Setting Parameters '''

# dictionary with all the configurations of meshes and the problem dimension

setup = {'dim': 5,
         'domain_sample_size': 4000 * 5,
         'boundary_sample_size': 40 * (5 ** 2)
         }


# for computational efficiency the functions will be evaluated here
def funcs(points, setup):
    xt = torch.Tensor(setup['domain_sample_size'], 0)

    for i in range(setup['dim'] + 1):
        xt = torch.cat((xt, points.interioru[i]), 1)

    h = func_h(points.init.to(device))
    f = func_f(xt.to(device))
    g = func_g(points.border.to(device))

    # this is meant to be a d by d-dimensional array containing domain_sample_size by 1 tensors
    a = torch.Tensor(setup['dim'], setup['dim'], setup['domain_sample_size'], 1)

    for i, j in product(range(setup['dim']), repeat=2):
        a[i, j] = func_a(xt, i, j)

    # this is meant to be a d-dimensional containing domain_sample_size by 1 tensors
    b = torch.Tensor(setup['dim'], setup['domain_sample_size'], 1)

    for i in range(setup['dim']):
        b[i] = func_b(xt, i)

    return h, f, g, a.to(device), b.to(device)


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
        inp = torch.Tensor(X[0].shape[0], 0).to(device)
        for i in range(setup['dim'] + 1):
            inp = torch.cat((inp, X[i]), 1)
        x = self.net(inp.view(-1, setup['dim'] + 1))
        return x

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


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
        inp = torch.Tensor(XV[0].shape[0], 0).to(device)
        for i in range(setup['dim'] + 1):
            inp = torch.cat((inp, XV[i]), 1)
        x = self.net(inp.view(-1, setup['dim'] + 1))
        return x

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return (self.loss)


# Hyperparameters

config = {
    'alpha': 1e4 * 40 * 25, #tune.loguniform(1e4, 1e8),   #
    'u_layers': 7,
    'u_hidden_dim': 20,
    'v_layers': 9,
    'v_hidden_dim': 50,
    'n1': 2,
    'n2': 1,
    'u_rate': 0.015,
    'v_rate': 0.04,
    'u_factor': 1 - 1e-10,
    'v_factor': 1 - 1e-10
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

    def __init__(self, border, alpha, a, b, h, f, g, setup, initialps, c=func_c, T=1, T0=0):
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
        self.initialps = initialps
        self.V = 1  # Volume of \Omega

    # TODO: check how the paper implements
    def I(self, y_output_u, y_output_v, ind, X, XV, u_net, v_net):
        y_output_u.retain_grad()
        y_output_v.retain_grad()
        N = y_output_u.shape[0]
        f = [func_w(XV[i]) for i in range(setup['dim'])]
        w = math.prod(f)
        phi = y_output_v * w
        [XV[i].retain_grad() for i in range(self.setup['dim'])]
        y_output_u.backward(torch.ones_like(y_output_u).to(device), retain_graph=True)
        du = {}
        for i in range(self.setup['dim']):
            du['du_' + str(i)] = X[i].grad
        phi.backward(torch.ones_like(phi).to(device), retain_graph=True)
        dphi = {}
        for i in range(self.setup['dim'] + 1):
            dphi['dphi_' + str(i)] = XV[i].grad
        fin = [self.initialps[ind * N:(ind + 1) * N, i].unsqueeze(1) for i in range(setup['dim'])]
        fin.append(self.T * torch.ones(fin[0].shape[0], 1).to(device))
        init = [self.initialps[ind * N:(ind + 1) * N, i].unsqueeze(1) for i in range(setup['dim'])]
        init.append(self.T0 * torch.ones(init[0].shape[0], 1).to(device))
        s1 = self.V * (u_net(fin) * v_net(fin) - self.h[ind * N:(ind + 1) * N].unsqueeze(1) * v_net(init)) / N
        s2 = (self.T - self.T0) * self.V * (y_output_u * dphi['dphi_2']) / N
        s31 = [self.a[i, j, ind * N:(ind + 1) * N, :] * dphi['dphi_' + str(i)] * du['du_' + str(j)] for i, j in
               product(range(self.setup['dim']), repeat=2)]
        s31 = np.sum(s31)
        s32 = np.sum([self.b[i, ind * N:(ind + 1) * N, :] * phi * du['du_' + str(i)] for i in range(self.setup['dim'])])
        c = self.c(y_output_u)
        s3f = s31 + s32 + c * y_output_u * phi + self.f[ind * N:(ind + 1) * N].unsqueeze(1) * phi
        s3c = (self.T - self.T0) * self.V / N
        s3 = s3c * s3f
        I = torch.sum(s1, 0) - torch.sum(s2 - s3, 0)
        [i.grad.data.zero_() for i in X]
        [i.grad.data.zero_() for i in XV]
        return I

    def init(self, u_net, ind):
        N = self.initialps.shape[0]
        init = [self.initialps[ind * N:(ind + 1) * N, i].unsqueeze(1) for i in range(setup['dim'])]
        init.append(self.T0 * torch.ones(init[0].shape[0], 1).to(device))
        return torch.mean((u_net(init) - self.h[ind * N:(ind + 1) * N]) ** 2)

    # initially had all variables feed with grad
    def bdry(self, ind, u_net, N):
        return torch.mean((u_net([self.boundary[ind * N:(ind + 1) * N, i].unsqueeze(1) for i in range(setup['dim'] + 1)])\
                           - self.g[ind * N:(ind + 1) * N].unsqueeze(1)) ** 2)

    def int(self, y_output_u, y_output_v, ind, X, XV, u_net, v_net):
        # x needs to be the set of points set plugged into net_u and net_v
        N = y_output_v.shape[0]
        return torch.log(self.I(y_output_u, y_output_v, ind, X, XV, u_net, v_net) ** 2) - torch.log(
            (self.T - self.T0) * self.V * torch.sum(y_output_v ** 2) / N)

    def u(self, y_output_u, y_output_v, ind, u_net, v_net, X, XV):
        N = y_output_u.shape[0]
        return self.int(y_output_u, y_output_v, ind, X, XV, u_net, v_net) + torch.mul((
                self.init(u_net, ind) + self.bdry(ind, u_net, N)), self.alpha)

    def v(self, y_output_u, y_output_v, ind, X, XV, u_net, v_net):
        return torch.mul(self.int(y_output_u, y_output_v, ind, X, XV, u_net, v_net), -1)


''' # Auxillary Funcs '''

def L_norm(X, predu, p):
    # p is the p in L^p
    xt = torch.Tensor(X[0].shape[0], 0)
    for i in X:
        xt = torch.cat((xt, i), 1)
    u_sol = func_u_sol(xt).to(device)
    return (torch.mean(torch.pow(torch.abs(u_sol - predu), p))) ** (1 / p)

def rel_err(X, predu):
    xt = torch.Tensor(X[0].shape[0], 0)
    for i in X:
        xt = torch.cat((xt, i), 1)
    u_sol = func_u_sol(xt).to(device)
    rel = torch.abs(torch.div(u_sol - predu, u_sol))
    return 100 * torch.mean(rel).item()

# TODO: change the proj function to make it work
def proj(u_net, axes=[0,1], down=-1, up=1, T=1, T0=0, n=setup['boundary_sample_size'], save=False):
    # Assumes hypercube
    assert len(axes) == 2, 'There can only be two axes in the graph to be able to display them'

    xt = torch.Tensor(n, setup['dim'] + 1)

    for i in axes:
        xt[:, i] = torch.Tensor(n).uniform_(down, up)

    for i in list(set(range(setup['dim'] + 1)) - set(axes)):
        xt[:, i] = up * torch.ones(n)

    if int(setup['dim']+1) in axes:
        xt[:, i] = torch.Tensor(n).uniform_(T0, T)

    u_sol = func_u_sol(xt).to(device)
    predu = u_net([xt[:, i].unsqueeze(1) for i in range(setup['dim'] + 1)]).to(device)

    error = predu-u_sol

    plt.clf()
    fig, ax = plt.subplots(3)
    aset = ax[0].contourf(xt[:, axes[0]].numpy(), xt[:, axes[1]].numpy(), u_sol.cpu().numpy())
    bset = ax[1].contourf(xt[:, axes[0]].numpy(), xt[:, axes[1]].numpy(), predu.cpu().numpy())
    cset = ax[2].contourf(xt[:, axes[0]].numpy(), xt[:, axes[1]].numpy(), error.cpu().numpy())
    fig.colorbar(aset, ax=ax[0])
    fig.colorbar(bset, ax=ax[1])
    fig.colorbar(cset, ax=ax[2])
    ax[0].set_title('Correct Solution, Guess and Error')
    plt.show()
    print('Displayed')

    if save:
        plt.savefig('plot_dim_' + str(axes[0]) + '_+_' + str(axes[1]) + '.png')
        print('Saved')


''' # Training '''

params = {**config, **setup, **{'iterations': int(2e4 + 1)}}

def train(params, checkpoint_dir=None):
    i = iter(params.items())
    config = dict(itertools.islice(i, 11))
    setup = dict(itertools.islice(i, 3))
    iterations = dict(i)['iterations']

    n1 = config['n1']
    n2 = config['n2']

    # neural network models
    u_net = torch.nn.DataParallel(generator(config)).to(device)
    v_net = torch.nn.DataParallel(discriminator(config)).to(device)

    u_net.apply(init_weights)
    v_net.apply(init_weights)

    for k in range(iterations):

        points = Comb_loader(setup['boundary_sample_size'], setup['domain_sample_size'],
                             setup['dim'])
        ds = DataLoader(points, num_workers=points.num_workers)

        h, f, g, a, b = funcs(points, setup)

        Loss = loss(points.border, config['alpha'], a, b, h, f, g, setup, points.init)

        # optimizers for WAN
        optimizer_u = torch.optim.Adam(u_net.parameters(), lr=config['u_rate'])
        optimizer_v = torch.optim.Adam(v_net.parameters(), lr=config['v_rate'])

        scheduler_u = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_u, factor=config['u_factor'], patience=100)
        scheduler_v = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_v, factor=config['v_factor'], patience=100)

        for i in range(n1):
            for ind, data in enumerate(ds):
                data = [data[i].squeeze(0).to(device).requires_grad_(True) for i in range(2 * setup['dim'] + 2)]
                X = data[:setup['dim'] + 1]
                XV = data[setup['dim'] + 1: 2 * setup['dim'] + 2]
                prediction_v = v_net(XV)
                prediction_u = u_net(X)
                loss_u = Loss.u(prediction_u, prediction_v, ind, u_net, v_net, X, XV)
                # writer.add_scalar("Loss", loss_u, k)
                optimizer_u.zero_grad()
                loss_u.backward(retain_graph=True)
                optimizer_u.step()
            scheduler_u.step(loss_u)

        for j in range(n2):
            for ind, data in enumerate(ds):
                data = [data[i].squeeze(0).to(device).requires_grad_(True) for i in range(2 * setup['dim'] + 2)]
                X = data[:setup['dim'] + 1]
                XV = data[setup['dim'] + 1: 2 * setup['dim'] + 2]
                prediction_v = v_net(XV)
                prediction_u = u_net(X)
                loss_v = Loss.v(prediction_u, prediction_v, ind, X, XV, u_net, v_net)
                optimizer_v.zero_grad()
                loss_v.backward(retain_graph=True)
                optimizer_v.step()
            scheduler_v.step(loss_v)

        if k % 10 == 0:
            lu, lv = loss_u.item(), loss_v.item()
            print('iteration: ' + str(k), 'Loss u: ' + str(lu), 'Loss v: ' + str(lv))
            X = points.interioru
            predu = u_net(X)
            L1 = L_norm(X, predu, 1).item()
            tune.report(L1=L1)
            print('L^1 norm ' + str(L1))
            print('Relative Error ' + str.format('{0:.3f}', rel_err(X, predu)) + '%')
            #proj(u_net, axes=[0, 5])


train(params)

'''
analysis = tune.run(
    train,
    num_samples=100,
    scheduler=ASHAScheduler(metric="L1", mode="min", grace_period=200, max_t=2e4),
    config=params,
    verbose=3,
    resources_per_trial={"cpu": 1, "gpu": 0.25},
    keep_checkpoints_num=1,
    checkpoint_score_attr="min-L1",
    resume=False
)

best_trial = analysis.get_best_trial(metric="L1", mode="min", scope='all')
'''

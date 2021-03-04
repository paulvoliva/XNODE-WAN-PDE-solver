# -*- coding: utf-8 -*-
"""PDE Solver.ipynb
Here we will take the time components in the third axis
"""

import math
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from Earlystop import EarlyStopping
from torch.autograd import grad
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

"""# Exact solution u(x) for the example PDE
we conduct a experiment on solving a IBVP with nonlinear diffusion-reaction equation and boundary condition involving time:
\begin{equation}
\left\{\begin{array}{ll}
u_{t}-\Delta u-u^{2}=f(x, y, t), & \text { in } \Omega \times[0, T] \\
u(x, y, t)=g(x, y, t), & \text { on } \partial \Omega \times[0, T] \\
u(x, y, 0)=h(x, y), & \text { in } \Omega
\end{array}\right.
\end{equation}
where $\Omega=(-1,1)^{2} \subset \mathbb{R}^{2}$. In this test, we give the definition of $f(x,y,t)=\left(\pi^{2}-2\right) \sin \left(\frac{\pi}{2} x\right) \cos \left(\frac{\pi}{2} y\right) e^{-t} - 4 \sin ^{2}\left(\frac{\pi}{2} x\right) \cos \left(\frac{\pi}{2} y\right) e^{-2 t}$ in $\Omega \times[0, T]$, $g(x,y,t)=2 \sin \left(\frac{\pi}{2} x\right) \cos \left(\frac{\pi}{2} y\right) e^{-t}$ on $\partial \Omega \times[0, T]$ and $h(x,y)=2 \sin \left(\frac{\pi}{2} x\right) \cos \left(\frac{\pi}{2} y\right)$ in $\Omega$. And the true solution is $u(x,y,t)=2 \sin \left(\frac{\pi}{2} x\right) \cos \left(\frac{\pi}{2} y\right) e^{-t}$.

# PDE Setup
"""

'''
# I use this to get the shape of the tensor when I debug
old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info
#'''

# TODO: check the structure for each function works and is doing what it is meant to
# TODO: compare the model with the source one

def func_u_sol(xt):
    l = xt.shape[0]
    u = 2 * torch.sin(math.pi / 2 * xt[:, 0, :]) * torch.cos(math.pi / 2 * xt[:, 1, :]) * torch.exp(xt[:, 2, :])
    return (u)


# We denote spatial coordinates with time as 'xt' and 'x' without

def func_f(xt):
    l = xt.shape[0]
    f = (math.pi ** 2 - 2) * torch.sin(math.pi / 2 * xt[:, 0, :]) * torch.cos(math.pi / 2 * xt[:, 1, :]) * torch.exp(
        -xt[:, 2, :]) - 4 * torch.sin(math.pi / 2 * xt[:, 0, :]) ** 2 * torch.cos(math.pi / 2 * xt[:, 1, :]) * torch.exp(-xt[:, 2, :])
    return (f)


def func_g(boundary_xt):
    return func_u_sol(boundary_xt)


def func_h(x):
    h = 2 * torch.sin(math.pi / 2 * x[:, 0]) * torch.cos(math.pi / 2 * x[:, 1])
    return h


def func_w(x):  # returns 1 for positions in the domain and 0 otherwise
    lens = x.shape[0]
    w_bool = torch.gt(1 - torch.abs(x[:, 0]), torch.zeros(lens)) & torch.gt(torch.abs(x[:, 0]), torch.zeros(lens))
    w_val = torch.where(w_bool, 1 - torch.abs(x[:, 0]) + torch.abs(x[:, 0]), torch.zeros(lens))
    return (w_val.view(-1, 1))


"""# Domain"""

T0 = 0  # if this is ignored it is always set as T0=0
T = 1

# Set up for a square
up = 1.0
down = -1.0
dim = 2
domain_sample_size = 5000
t_mesh_size = 11
boundary_sample_size = 500

# defining the training domain
x0_domain = torch.Tensor(domain_sample_size, dim).uniform_(down, up)
x0_domain.requires_grad_()

x_domain_train = x0_domain.unsqueeze(2).repeat(1, 1, t_mesh_size)

t = torch.linspace(T0, T, t_mesh_size).unsqueeze(1).unsqueeze(2).view(1, 1, t_mesh_size).repeat(domain_sample_size, 1, 1)
xt_domain_train = torch.cat((x_domain_train, t), dim=1)

# defining the training boundary
x0_boundary_side = torch.Tensor(boundary_sample_size, dim - 1).uniform_(down, up)
x0_boundary_side.requires_grad_()

x0_boundary_left = torch.cat((torch.ones(x0_boundary_side.size()) * down, x0_boundary_side), 1)
x0_boundary_right = torch.cat((torch.ones(x0_boundary_side.size()) * up, x0_boundary_side), 1)
x0_boundary_down = torch.cat((x0_boundary_side, torch.ones(x0_boundary_side.size()) * down), 1)
x0_boundary_up = torch.cat((x0_boundary_side, torch.ones(x0_boundary_side.size()) * up), 1)

x0_boundary = torch.cat((x0_boundary_left, x0_boundary_right, x0_boundary_down, x0_boundary_up), 0)

x_boundary_train = x0_boundary.unsqueeze(2).repeat(1, 1, t_mesh_size)
xt_boundary_train = torch.cat((x_boundary_train, t[:4*boundary_sample_size, :, :]), dim=1)



# Validation data Sets
val_domain_size = int(domain_sample_size * 0.3)
val_boundary_size = int(boundary_sample_size * 0.3)

x0_domain_val = torch.Tensor(val_domain_size, dim).uniform_(down, up)
x0_domain_val.requires_grad_()

x_domain_val = x0_domain_val.unsqueeze(2).repeat(1, 1, t_mesh_size)
xt_domain_val = torch.cat((x_domain_val, t[:val_domain_size, :, :]), dim=1)

# defining the validation boundary
x0_boundary_side = torch.Tensor(val_boundary_size, dim - 1).uniform_(down, up)
x0_boundary_side.requires_grad_()

x0_boundary_left = torch.cat((torch.ones(x0_boundary_side.size()) * down, x0_boundary_side), 1)
x0_boundary_right = torch.cat((torch.ones(x0_boundary_side.size()) * up, x0_boundary_side), 1)
x0_boundary_down = torch.cat((x0_boundary_side, torch.ones(x0_boundary_side.size()) * down), 1)
x0_boundary_up = torch.cat((x0_boundary_side, torch.ones(x0_boundary_side.size()) * up), 1)

x0_boundary = torch.cat((x0_boundary_left, x0_boundary_right, x0_boundary_down, x0_boundary_up), 0)

x_boundary_val = x0_boundary.unsqueeze(2).repeat(1, 1, t_mesh_size)
xt_boundary_val = torch.cat((x_boundary_val, t[:4*val_boundary_size, :, :]), dim=1)

xv = xt_domain_train[:, 0, :].clone().detach()
yv = xt_domain_train[:, 1, :].clone().detach()
tv = xt_domain_train[:, 2, :].clone().detach()
xv = xv.requires_grad_(True)
yv = yv.requires_grad_(True)
tv = tv.requires_grad_(True)


xu = xv.clone().detach()
yu = yv.clone().detach()
tu = tv.clone().detach()
xu.requires_grad_(True)
yu.requires_grad_(True)
tu.requires_grad_(True)

x_val = xt_domain_val[:, 0, :].clone().detach().requires_grad_(True)
y_val = xt_domain_val[:, 1, :].clone().detach().requires_grad_(True)
t_val = xt_domain_val[:, 2, :].clone().detach().requires_grad_(True)

"""# Defining the Model"""


class generator(torch.nn.Module):  # this makes the u function
    def __init__(self, config):
        super().__init__()
        self.num_layers = config['u_layers']
        self.hidden_dim = config['u_hidden_dim']
        self.input = torch.nn.Linear(dim+1, self.hidden_dim)
        self.hidden = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = torch.nn.Linear(self.hidden_dim, 1)
        self.net = torch.nn.Sequential(*[
            self.input,
            *[torch.nn.ReLU(), self.hidden] * self.num_layers,
            torch.nn.Tanh(),
            self.output

        ])

    def forward(self, x0, y0, t):
        #t = t.unsqueeze(1).unsqueeze(2).view(1, 1, t_mesh_size).repeat(1, 2, 1)
        #x_ = torch.cat((x0.unsqueeze(2).view(-1, 1, t_mesh_size), y0.unsqueeze(2).view(-1, 1, t_mesh_size)), dim=1)
        inp = torch.cat((x0.unsqueeze(2), y0.unsqueeze(2), t.unsqueeze(2)), dim=2)
        x = self.net(inp)
        return x

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return (self.loss)


class discriminator(torch.nn.Module):  # this makes the v function
    def __init__(self, config):
        super().__init__()
        self.num_layers = config['v_layers']
        self.hidden_dim = config['v_hidden_dim']
        self.input = torch.nn.Linear(dim+1, self.hidden_dim)
        self.hidden = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = torch.nn.Linear(self.hidden_dim, 1)
        self.net = torch.nn.Sequential(*[
            self.input,
            *[torch.nn.ReLU(), self.hidden] * self.num_layers,
            torch.nn.Tanh(),
            self.output

        ])

    def forward(self, x0, y0, t):
        #t = t.unsqueeze(1).unsqueeze(2).view(1, 1, t_mesh_size).repeat(1, 2, 1)
        #x_ = torch.cat((x0.unsqueeze(2).view(-1, 1, t_mesh_size), y0.unsqueeze(2).view(-1, 1, t_mesh_size)), dim=1)
        inp = torch.cat((x0.unsqueeze(2), y0.unsqueeze(2), t.unsqueeze(2)), dim=2)
        x = self.net(inp)
        return x

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return (self.loss)

config = {
    'u_layers': 6,
    'u_hidden_dim': 20,
    'v_layers': 6,
    'v_hidden_dim': 50,
    'n1': 10,
    'n2': 6,
    'u_rate': 0.00015,
    'v_rate': 0.00015}

'''
config = {
    'u_layers': 5,
    'u_hidden_dim': 20,
    'v_layers': 6,
    'v_hidden_dim': 16,
    'n1': 10,
    'n2': 6,
    'u_rate': 0.0025703,
    'v_rate': 0.00593051}


config = {
    'u_layers': tune.choice([4, 5, 6, 7, 8]),
    'u_hidden_dim': tune.choice([20, 21, 23]),
    'v_layers': tune.choice([5, 6, 7, 8]),
    'v_hidden_dim': tune.choice([13, 15, 16]),
    'n1': 10,
    'n2': 5,
    'u_rate': tune.loguniform(1e-3, 1e-2),
    'v_rate': tune.loguniform(1e-3, 1e-2)
}
'''

"""# Loss Function"""

# look into gradients
def I(y_output_u, y_output_v, xt, xv, yv, tv, xu, yu, tu):
    shape = [y_output_u.shape[0], t_mesh_size, dim]
    shape[-1] = shape[-1] - 1
    y_output_u.retain_grad()
    y_output_v.retain_grad()
    phi = y_output_v * func_w(xt[:, :, 0]).unsqueeze(2).repeat(1, 11, 1)
    y_output_u.backward(torch.ones(shape), retain_graph=True)
    du_x = xu.grad
    du_y = yu.grad
    phi.backward(torch.ones(shape), retain_graph=True)
    dphi_x = xv.grad
    dphi_y = yv.grad
    dphi_t = tv.grad.unsqueeze(2)
    # TODO: go through the loss functions to ensure they are correct
    s1 = y_output_u[:, -1, :] * phi[:, -1, :] - func_h(xt[:, 0, :]).unsqueeze(1) * phi[:, 0, :]
    s2 = y_output_u * dphi_t  # for t does this make sense?
    Lap = du_x * dphi_x + du_y * dphi_y
    s3 = Lap.unsqueeze(2) + y_output_u * y_output_u * phi - func_f(xt).unsqueeze(2) * phi
    I = torch.sum(s1 - torch.sum(s2 - s3, 1), 0)
    xu.grad.data.zero_()
    yu.grad.data.zero_()
    xv.grad.data.zero_()
    yv.grad.data.zero_()
    tv.grad.data.zero_()
    return I
# TODO: compare them with the source material

def L_init(y_output_u, config):
    #u_net = generator(config)
    return torch.mean((y_output_u[:, 0, 0] - func_h(xt_domain_train[:, :2, 0])) ** 2)


def L_bdry(u_net):
    return torch.mean((u_net(xt_boundary_train[:, 0, :], xt_boundary_train[:, 1, :], xt_boundary_train[:, 2, :]) -
                       func_g(xt_boundary_train).unsqueeze(2)) ** 2)


def L_int(y_output_u, y_output_v, xt=xt_domain_train, xv=xv, yv=yv, tv=tv, xu=xu, yu=yu, tu=tu):
    # x needs to be the set of points set plugged into net_u and net_v
    return torch.log((I(y_output_u, y_output_v, xt, xv, yv, tv, xu, yu, tu)) ** 2) - torch.log(torch.sum(y_output_v ** 2))


gamma = 25
alpha = 25


def L(y_output_u, y_output_v, u_net):
    return L_int(y_output_u, y_output_v) + gamma * L_init(y_output_u, config) + alpha * L_bdry(
        u_net)


def Loss_u(y_output_u, y_output_v, u_net):
    return L(y_output_u, y_output_v, u_net)


def Loss_v(y_output_u, y_output_v):
    return -L_int(y_output_u, y_output_v)


"""# Training"""

iteration = 101

def train(config):
    n1 = config['n1']
    n2 = config['n2']

    # neural network models
    u_net = generator(config)
    v_net = discriminator(config)

    # optimizers for WAN
    optimizer_u = torch.optim.Adam(u_net.parameters(), lr=config['u_rate'])
    optimizer_v = torch.optim.Adam(v_net.parameters(), lr=config['v_rate'])

    #scheduler_u = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_u, factor=0.5, patience=30)
    #scheduler_v = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_v, factor=0.5, patience=30)

    prediction_u = u_net(xu, yu, tu)
    prediction_v = v_net(xv, yv, tv)

    for k in range(iteration):

        for i in range(n1):
            loss_u = Loss_u(prediction_u, prediction_v, u_net)
            optimizer_u.zero_grad()
            loss_u.backward(retain_graph=True)
            optimizer_u.step()
            #scheduler_u.step(loss_u)
            prediction_u = u_net(xu, yu, tu)

        for j in range(n2):
            loss_v = Loss_v(prediction_u, prediction_v)
            optimizer_v.zero_grad()
            loss_v.backward(retain_graph=True)
            optimizer_v.step()
            #scheduler_v.step(loss_v)
            prediction_v = v_net(xv, yv, tv)


        if k % 10 == 0:
            torch.save(u_net.state_dict(), "./net_u.pth")
            torch.save(v_net.state_dict(), "./net_v.pth")
            print(k, loss_u.data, loss_v.data)
            #print('learning rate at %d epoch：%f' % (k, optimizer_u.param_groups[0]['lr']))
            #print('learning rate at %d epoch：%f' % (k, optimizer_v.param_groups[0]['lr']))
            error_test = torch.mean(
                torch.sqrt(torch.square((func_u_sol(xt_domain_train) - prediction_u.data.squeeze(2))))).data
            print("error test " + str(error_test))
        #error_test = torch.mean(torch.sqrt(torch.square((func_u_sol(xt_domain_train) - prediction_u.data.squeeze(2))))).data
        error_test = torch.mean(torch.sqrt(torch.square(func_u_sol(xt_domain_val)-u_net(x_val, y_val, t_val).data.squeeze(2))))
        #print(error_test)
        #tune.report(mean_error=float(error_test.detach().numpy()))


        #EarlyStopping(loss_u, u_net)


train(config)

'''
# Hyperparameter Optimization

# Best trial config: {'u_layers': 5, 'u_hidden_dim': 20, 'v_layers': 5, 'v_hidden_dim': 5, 'n1': 10, 'n2': 5, 'u_rate': 0.0002133670132946601, 'v_rate': 0.0006349874886164373}
# Best trial config: {'u_layers': 2, 'u_hidden_dim': 20, 'v_layers': 4, 'v_hidden_dim': 20, 'n1': 10, 'n2': 7, 'u_rate': 0.0006756995220370141, 'v_rate': 0.0002804879090959305}
# Best trial config: {'u_layers': 4, 'u_hidden_dim': 23, 'v_layers': 4, 'v_hidden_dim': 15, 'n1': 10, 'n2': 6, 'u_rate': 0.003370553415547106, 'v_rate': 0.009087847200586583}

analysis = tune.run(
    train,
    num_samples=100,
    scheduler=ASHAScheduler(metric="mean_error", mode="min"),
    config=config,
    verbose=3)

best_trial = analysis.get_best_trial(metric="mean_error", mode="min")
print("Best trial config: {}".format(best_trial.config))


# Obtain a trial dataframe from all run trials of this `tune.run` call.
dfs = analysis.trial_dataframes


ax = None  # This plots everything on the same plot
for d in dfs.values():
    ax = d.mean_error.plot(ax=ax, legend=False)
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean Error")
'''

x_mesh = torch.linspace(down, up, 500, requires_grad=True)
mesh1, mesh2 = torch.meshgrid(x_mesh, x_mesh)
mesh_1 = torch.reshape(mesh1, [-1, 1]).repeat(1, 11).unsqueeze(2).view(-1, 1, 11)
mesh_2 = torch.reshape(mesh2, [-1, 1]).repeat(1, 11).unsqueeze(2).view(-1, 1, 11)
t = torch.linspace(0, 1, 11).unsqueeze(1).view(1, -1).repeat(250000, 1).unsqueeze(2).view(-1, 1, 11)
xt_comparison = torch.cat((mesh_1, mesh_2, t), dim=1)

u_net = generator(config)

error = torch.sqrt(torch.square(func_u_sol(xt_comparison)-u_net(xt_comparison[:, 0, :], xt_comparison[:, 1, :], xt_comparison[:, 2, :]).squeeze(2)))
mean_error = torch.mean(error, dim=1)
mean_error = mean_error.view(500, 500)

plt.figure(figsize=(10,6))

cset=plt.contourf(mesh1.data.numpy(),mesh2.data.numpy(),mean_error.data.numpy(), 500, cmap='winter')

plt.colorbar(cset)
plt.show()

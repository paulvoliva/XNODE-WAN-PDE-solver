# -*- coding: utf-8 -*-
"""PDE Solver.ipynb
Here we will take the time components in the third axis
"""
import json
import math
import torch
import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
# from Earlystop import EarlyStopping
from torch.autograd import grad
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import datetime
from itertools import product
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
print(os.environ["CUDA_VISIBLE_DEVICES"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')
# TODO: implement cuda

def func_u_sol(xt):
    l = xt.shape[0]
    u = 2 * torch.sin(math.pi / 2 * xt[:, 0, :]) * torch.cos(math.pi / 2 * xt[:, 1, :]) * torch.exp(-xt[:, 2, :])
    return(u)


# We denote spatial coordinates with time as 'xt' and 'x' without

def func_f(xt):
    l = xt.shape[0]
    f = (math.pi ** 2 - 2) * torch.sin(math.pi / 2 * xt[:, 0, :]) * torch.cos(math.pi / 2 * xt[:, 1, :]) * torch.exp(
        -xt[:, 2, :]) - 4 * torch.sin(math.pi / 2 * xt[:, 0, :]) ** 2 * (torch.cos(math.pi / 2 * xt[:, 1, :]) **2)* torch.exp(-xt[:, 2, :])
    return(f)


def func_g(boundary_xt):
    return func_u_sol(boundary_xt)


def func_h(x):
    h = 2 * torch.sin(math.pi / 2 * x[:, 0]) * torch.cos(math.pi / 2 * x[:, 1])
    return h

def func_c(y_output_u):
    return -y_output_u

def func_w(x):  # returns 1 for positions in the domain and 0 otherwise
    lens = x.shape[0]
    w_bool = torch.gt(1 - torch.abs(x[:, 0]), torch.zeros(lens).to(device)) & torch.gt(torch.abs(x[:, 0]), torch.zeros(lens).to(device))
    w_val = torch.where(w_bool, 1 - torch.abs(x[:, 0]) + torch.abs(x[:, 0]), torch.zeros(lens).to(device))
    return (w_val.view(-1, 1))


"""# Domain"""

T0 = 0  # if this is ignored it is always set as T0=0
T = 1

# Set up for a square
up = 1.0
down = -1.0
dim = 2
domain_sample_size =1000  # 8000, 25000
t_mesh_size = 11

boundary_sample_size = domain_sample_size //25

# defining the training domain
x0_domain = torch.Tensor(domain_sample_size, dim).uniform_(down, up)
x0_domain.requires_grad_()

x_domain_train = x0_domain.unsqueeze(2).repeat(1, 1, t_mesh_size)

t = torch.linspace(T0, T, t_mesh_size).unsqueeze(1).unsqueeze(2).view(1, 1, t_mesh_size).repeat(domain_sample_size, 1, 1)
xt_domain_train = torch.cat((x_domain_train, t), dim=1).to(device)


# defining the training boundary
x0_boundary_side = torch.Tensor(boundary_sample_size, dim - 1).uniform_(down, up)
x0_boundary_side.requires_grad_()

x0_boundary_left = torch.cat((torch.ones(x0_boundary_side.size()) * down, x0_boundary_side), 1)
x0_boundary_right = torch.cat((torch.ones(x0_boundary_side.size()) * up, x0_boundary_side), 1)
x0_boundary_down = torch.cat((x0_boundary_side, torch.ones(x0_boundary_side.size()) * down), 1)
x0_boundary_up = torch.cat((x0_boundary_side, torch.ones(x0_boundary_side.size()) * up), 1)

x0_boundary = torch.cat((x0_boundary_left, x0_boundary_right, x0_boundary_down, x0_boundary_up), 0)

x_boundary_train = x0_boundary.unsqueeze(2).repeat(1, 1, t_mesh_size)
xt_boundary_train = torch.cat((x_boundary_train, t[:4*boundary_sample_size, :, :]), dim=1).to(device)

# Validation data Sets
val_domain_size = int(domain_sample_size * 0.3)
val_boundary_size = int(boundary_sample_size * 0.3)

x0_domain_val = torch.Tensor(val_domain_size, dim).uniform_(down, up)
x0_domain_val.requires_grad_()

x_domain_val = x0_domain_val.unsqueeze(2).repeat(1, 1, t_mesh_size)
xt_domain_val = torch.cat((x_domain_val, t[:val_domain_size, :, :]), dim=1).to(device)

# defining the validation boundary
x0_boundary_side = torch.Tensor(val_boundary_size, dim - 1).uniform_(down, up)
x0_boundary_side.requires_grad_()

x0_boundary_left = torch.cat((torch.ones(x0_boundary_side.size()) * down, x0_boundary_side), 1)
x0_boundary_right = torch.cat((torch.ones(x0_boundary_side.size()) * up, x0_boundary_side), 1)
x0_boundary_down = torch.cat((x0_boundary_side, torch.ones(x0_boundary_side.size()) * down), 1)
x0_boundary_up = torch.cat((x0_boundary_side, torch.ones(x0_boundary_side.size()) * up), 1)

x0_boundary = torch.cat((x0_boundary_left, x0_boundary_right, x0_boundary_down, x0_boundary_up), 0)

x_boundary_val = x0_boundary.unsqueeze(2).repeat(1, 1, t_mesh_size)
xt_boundary_val = torch.cat((x_boundary_val, t[:4*val_boundary_size, :, :]), dim=1).to(device)

###################################
xv = xt_domain_train[:, 0, :].clone().detach()
yv = xt_domain_train[:, 1, :].clone().detach()
tv = xt_domain_train[:, 2, :].clone().detach()
xv = xv.requires_grad_(True).to(device)
yv = yv.requires_grad_(True).to(device)
tv = tv.requires_grad_(True).to(device)



xu = xv.clone().detach()
yu = yv.clone().detach()
tu = tv.clone().detach()
xu.requires_grad_(True).to(device)
yu.requires_grad_(True).to(device)
tu.requires_grad_(True).to(device)



X = [xu, yu, tu]
XV = [xv, yv, tv]

x_val = xt_domain_val[:, 0, :].clone().detach().requires_grad_(True).to(device)
y_val = xt_domain_val[:, 1, :].clone().detach().requires_grad_(True).to(device)
t_val = xt_domain_val[:, 2, :].clone().detach().requires_grad_(True).to(device)

###newly added
a1, a2 = torch.cat((torch.ones(1, 1, domain_sample_size, t_mesh_size, 1), torch.zeros(1, 1, domain_sample_size, t_mesh_size, 1)), dim=1), torch.cat((torch.zeros(1, 1, domain_sample_size, t_mesh_size, 1), torch.ones(1, 1, domain_sample_size, t_mesh_size, 1)), dim=1)
a = torch.cat((a1, a2), dim=0).to(device)

# this is meant to be a d-dimensional containing domain_sample_size by t_mesh_size by 1 tensors
b = torch.cat((torch.zeros(1, domain_sample_size, t_mesh_size, 1), torch.zeros(1, domain_sample_size, t_mesh_size, 1)), dim=0).to(device)

x_setup = xv.clone().detach().to(device)
y_setup = yv.clone().detach().to(device)
t_setup = tv.clone().detach().to(device)

xyt_setup = torch.cat((x_setup.unsqueeze(2).view(-1, 1, t_mesh_size), y_setup.unsqueeze(2).view(-1, 1, t_mesh_size), t_setup.unsqueeze(2).view(-1, 1, t_mesh_size)), dim=1)

h = func_h(xyt_setup[:, :, 0]).to(device)
f = func_f(xyt_setup).to(device)
g = func_g(xt_boundary_train.clone().detach().to(device)).unsqueeze(2).to(device)


"""# Defining the Model"""

def init_weights(layer):
    if type(layer) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)

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
        inp = torch.cat((x0.unsqueeze(2), y0.unsqueeze(2), t.unsqueeze(2)), dim=2)
        x = self.net(inp)
        return x



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
        inp = torch.cat((x0.unsqueeze(2), y0.unsqueeze(2), t.unsqueeze(2)), dim=2)
        x = self.net(inp)
        return x


#Serach space based on the WAN Paper
config = {
    'alpha': tune.loguniform(1e1, 1e5),
    #'gamma': tune.loguniform(1e1, 1e5), #tune.uniform(10, 50),#tune.loguniform(1e1, 1e5),#tune.loguniform(1e3, 1e7),
    'u_layers': 7,
    'u_hidden_dim': 20,
    'v_layers': 7,
    'v_hidden_dim': 50,
    'n1': 10,  # tune.choice([2, 4, 6, 8, 10, 14, 20]),
    'n2': 5,  # tune.sample_from(lambda spec: int(spec.config.n1/2)),
    'u_rate': tune.loguniform(0.001, 0.1),
    'v_rate': tune.loguniform(0.001, 0.1),
    'iteration': 4000,
    'subiteration':10,
    'u_factor': tune.loguniform(0.3,0.99999),
    'v_factor':tune.loguniform(0.3,0.99999)
}

#     'u_rate': tune.loguniform(0.001, 0.1),
#     'v_rate': tune.loguniform(0.001, 0.1)
# ------+------------+-----------|
# | train_57c1d_00000 | RUNNING  |       | 0.00420757 | 0.0063911 |

#     'u_rate': tune.loguniform(0.002, 0.1),
#     'v_rate': tune.loguniform(0.001, 0.1)
# | train_9a821_00000 | RUNNING  |       | 0.0160603 | 0.00142824
'''
#Original WAN paper
config = {
    'u_layers': 7,
    'u_hidden_dim': 20,
    'v_layers': 7,
    'v_hidden_dim': 50,
    'n1': 10,  # 2,
    'n2': 5,  # 6, 1
    'u_rate': 0.015,  # 0.00015
    'v_rate': 0.04   # 0.00015
}


#One hyperparameter result
config = {
    'u_layers': 5,
    'u_hidden_dim': 20,
    'v_layers': 6,
    'v_hidden_dim': 16,
    'n1': 10,
    'n2': 6,
    'u_rate': 0.0025703,
    'v_rate': 0.00593051}

#Search Space for the above
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

def I(y_output_u, y_output_v, XV, X, a=a, b=b,h=h, f=f, c=func_c):
    y_output_u.retain_grad()
    y_output_v.retain_grad()
    phi = y_output_v * func_w(xv).unsqueeze(2).repeat(1, t_mesh_size, 1)
    y_output_u.backward(torch.ones_like(y_output_u).to(device), retain_graph=True)
    du = {}
    for i in range(dim):
        du['du_'+str(i)] = X[i].grad
    phi.backward(torch.ones_like(phi).to(device), retain_graph=True)
    dphi = {}
    for i in range(dim+1):
        dphi['dphi_'+str(i)] = XV[i].grad
    s1 = y_output_u[:, -1, :].squeeze(1) * phi[:, -1, :].squeeze(1) - h
    s2 = (y_output_u * dphi['dphi_2'].unsqueeze(2))/t_mesh_size  # for t does this make sense?
    s31 = 0
    for i,j in product(range(dim), repeat=2):
        s31 += a[i, j, :, :, :] * dphi['dphi_'+str(i)].unsqueeze(2) * du['du_'+str(j)].unsqueeze(2)
    s32 = 0
    for i in range(dim):
        s32 += b[i] * phi * du['du_'+str(i)].unsqueeze(2)
    s3 = (T-T0)*(s31 + s32 + func_c(y_output_u) * y_output_u * phi - f.unsqueeze(2) * phi)/t_mesh_size
    I = torch.sum(s1 - torch.sum(s2 - s3, 1).squeeze(1), 0)
    for i in X:
        i.grad.data.zero_()
    for i in XV:
        i.grad.data.zero_()
    return I

# TODO: ensure loss is normalised
# def I(y_output_u, y_output_v, xt, xv, yv, tv, xu, yu, tu):
#     shape = [y_output_u.shape[0], t_mesh_size, dim]
#     shape[-1] = shape[-1] - 1
#     y_output_u.retain_grad()
#     y_output_v.retain_grad()
#     phi = y_output_v * func_w(xt[:, :, 0]).unsqueeze(2).repeat(1, t_mesh_size, 1)
#     y_output_u.backward(torch.ones(shape).to(device), retain_graph=True)
#     du_x = xu.grad
#     du_y = yu.grad
#     phi.backward(torch.ones(shape).to(device), retain_graph=True)
#     dphi_x = xv.grad
#     dphi_y = yv.grad
#     dphi_t = tv.grad.unsqueeze(2)
#     s1 = y_output_u[:, -1, :] * phi[:, -1, :] - func_h(xt[:, 0, :]).unsqueeze(1) * phi[:, 0, :]
#     s2 = (y_output_u * dphi_t)/t_mesh_size  # for t does this make sense?
#     Lap = du_x * dphi_x + du_y * dphi_y
#     s3 = (T-T0)*(Lap.unsqueeze(2) - y_output_u * y_output_u * phi - func_f(xt).unsqueeze(2) * phi)/t_mesh_size
    
#       # From Paul:  I also realised I made a typo in my loss function where I added 'y_output_u * y_output_u * phi' instead of subtracting it. Then I also noted that 'alpha' directly influences the size of the loss function so that even though a large 'alpha' may be good it will necessarily produce a large loss value and so the hyperparameter tuner will think its bad. I don't know how to solve this without getting the reverse effect (as only two terms in the sum have alpha as a coefficient). I am very sorry for my mistakes that have caused a big waste of time. I hope this will not happen again in the future.

#     I = torch.sum(s1 - torch.sum(s2 - s3, 1), 0)
#     xu.grad.data.zero_()
#     yu.grad.data.zero_()
#     xv.grad.data.zero_()
#     yv.grad.data.zero_()
#     tv.grad.data.zero_()
#     return I

# def L_init(y_output_u):
#     return torch.mean((y_output_u[:, 0, 0] - func_h(xt_domain_train[:, :2, 0])) ** 2)

def L_init(y_output_u, h=h):
    return torch.mean((y_output_u[:, 0, :].squeeze(1) - h) ** 2)

def L_bdry(u_net, g=g):
    return torch.mean((u_net(xt_boundary_train[:, 0, :], xt_boundary_train[:, 1, :], xt_boundary_train[:, 2, :]) - g) ** 2)

# def L_bdry(u_net):
#     return torch.mean((u_net(xt_boundary_train[:, 0, :], xt_boundary_train[:, 1, :], xt_boundary_train[:, 2, :]) -
#                        func_g(xt_boundary_train).unsqueeze(2)) ** 2)

def L_int(y_output_u, y_output_v, XV=XV, X=X):
    # x needs to be the set of points set plugged into net_u and net_v
    return torch.log((I(y_output_u, y_output_v, XV, X)) ** 2) - torch.log(torch.sum(y_output_v ** 2))

# def L_int(y_output_u, y_output_v, xt=xt_domain_train, xv=xv, yv=yv, tv=tv, xu=xu, yu=yu, tu=tu):
#     # x needs to be the set of points set plugged into net_u and net_v
#     return torch.log((I(y_output_u, y_output_v, xt, xv, yv, tv, xu, yu, tu)) ** 2) - torch.log(torch.sum(y_output_v ** 2))


#gamma = 25    # 1e5*boundary_sample_size*4  # 25
#alpha = gamma


# def L(y_output_u, y_output_v, u_net, alpha, gamma):
#     return L_int(y_output_u, y_output_v) + gamma * L_init(y_output_u) + alpha * L_bdry(
#         u_net)


# def Loss_u(y_output_u, y_output_v, u_net, alpha, gamma):
#     return L(y_output_u, y_output_v, u_net, alpha, gamma)


# def Loss_v(y_output_u, y_output_v):
#     return -L_int(y_output_u, y_output_v)

def Loss_u(y_output_u, y_output_v, u_net, alpha, gamma):
    return L_int(y_output_u, y_output_v) + gamma * L_init(y_output_u) + alpha * L_bdry(u_net)

def Loss_v(y_output_u, y_output_v):
    return -L_int(y_output_u, y_output_v)

# TODO: identify the best n1, n2 (n2=n1/2) and best alpha and gamma

"""# Training"""


x_mesh = torch.linspace(0, 1, 50, requires_grad=True)
mesh1, mesh2 = torch.meshgrid(x_mesh, x_mesh)
mesh_1= torch.reshape(mesh1, [-1,1])
mesh_2= torch.reshape(mesh2, [-1,1])

t = torch.ones(2500, 1)
xt_test = torch.cat((mesh_1, mesh_2, t), dim = 1).unsqueeze(2)

xt_test.to(device)

def train(config, checkpoint_dir=None):
    n1 = config['n1']
    n2 = config['n2']
    
    iteration=config['iteration']
    nn=config['subiteration']
    
    # neural network models
    u_net = generator(config).to(device)
    v_net = discriminator(config).to(device)
    
    u_net.apply(init_weights)
    v_net.apply(init_weights)

    #scheduler_u = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_u, factor=0.5, patience=30)
    #scheduler_v = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_v, factor=0.5, patience=30)

    prediction_u = u_net(xu, yu, tu)
    prediction_v = v_net(xv, yv, tv)

    Loss = 0
    
            
    for k in range(iteration):
        
        # optimizers for WAN
        optimizer_u = torch.optim.Adam(u_net.parameters(), lr=config['u_rate'])
        optimizer_v = torch.optim.Adam(v_net.parameters(), lr=config['v_rate'])
        
        scheduler_u = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_u, factor=config['u_factor'], patience=0)
        scheduler_v = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_v, factor=config['v_factor'], patience=0)
        
#         if k==0:
#                 if checkpoint_dir:
#                     with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
#                     state = json.loads(f.read())
#                     start = state["step"] + 1
        
        for i in range(n1):
            
            loss_u = Loss_u(prediction_u, prediction_v, u_net, config['alpha'], config['alpha'])
            
            optimizer_u.zero_grad()
            loss_u.backward(retain_graph=True)
            try:
                optimizer_u.step()
            except TypeError:
                print("this opt_u step is wrong")
            scheduler_u.step(loss_u)
            prediction_u = u_net(xu, yu, tu)            

        for j in range(n2):
            loss_v = Loss_v(prediction_u, prediction_v)
            
            optimizer_v.zero_grad()
            loss_v.backward(retain_graph=True)
            try:
                optimizer_v.step()
            except TypeError:
                print("this opt_v step is wrong")
                
            scheduler_v.step(loss_v)
            prediction_v = v_net(xv, yv, tv)
#             print('v',k,i,torch.isnan(prediction_v).any()) #sanity check
        
        loss_u = Loss_u(prediction_u, prediction_v, u_net, 1, 1)        #this is to ensure that for our reported losses they are all scaled equally to avoid a bias away from large alpha
        Loss += 0.1*loss_u

        if k % nn == 0:
            #torch.save(u_net.state_dict(), "./net_u.pth")
            #torch.save(v_net.state_dict(), "./net_v.pth")
            print(k, loss_u.data.item(), loss_v.data.item())
            # print('learning rate at %d epoch：%f' % (k, optimizer_u.param_groups[0]['lr']))
            # print('learning rate at %d epoch：%f' % (k, optimizer_v.param_groups[0]['lr']))
            
            tune.report(loss=Loss.item())
            Loss = 0
            error_test = torch.mean(
                torch.sqrt(torch.square((func_u_sol(xt_domain_train) - prediction_u.data.squeeze(2))))).data
            print("error test " + str(error_test))
            with tune.checkpoint_dir(k) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((u_net.state_dict(), v_net.state_dict()), path)
                
#             with tune.checkpoint_dir(step=k) as checkpoint_dir:
#                 path = os.path.join(checkpoint_dir, "checkpoint")
#                 torch.save((u_net.state_dict(), v_net.state_dict()), path)     
                       
#             tune.report(loss=Loss.item())          
#             Loss = 0

#             error_test = torch.mean(
#                 torch.sqrt(torch.square((func_u_sol(xt_domain_train) - prediction_u.data.squeeze(2))))).data
#             print("Error test " + str(error_test.item()))
            
                
        del loss_u, loss_v
        torch.cuda.empty_cache()
            #PATHg = os.path.join('/Users/paulvalsecchi/Desktop/Project/Models alpha=25', 'generator_{}.pth'.format(k))
            #PATHd = os.path.join('/Users/paulvalsecchi/Desktop/Project/Models alpha=25', 'discriminator_{}.pth'.format(k))

            #torch.save(u_net.state_dict(), PATHg)
            #torch.save(v_net.state_dict(), PATHd)


        #error_test = torch.mean(torch.sqrt(torch.square((func_u_sol(xt_domain_train) - prediction_u.data.squeeze(2))))).data
#         error_test = torch.mean(torch.sqrt(torch.square(func_u_sol(xt_domain_val)-u_net(x_val, y_val, t_val).data.squeeze(2)))) newly deleted
#         tune.report(loss=error_test.data) 
        #tune.report(Loss=float(loss_u.detach().numpy()))
        


        #EarlyStopping(loss_u, u_net)
        
   

'''
train(config)

u_net = generator(config)
PATHg = os.path.join('/Users/paulvalsecchi/Desktop/Project/Models', 'generator_1000.pth')
u_net.load_state_dict(torch.load(PATHg))
u_net.eval()

plt.plot(func_u_sol(xt_test).data.numpy())
plt.plot(u_net(xt_test[:, 0, 0].unsqueeze(1), xt_test[:, 1, 0].unsqueeze(1), xt_test[:, 2, 0].unsqueeze(1)).squeeze(2).data.numpy())
plt.show()
'''

# Hyperparameter Optimization

# Best trial config: {'u_layers': 5, 'u_hidden_dim': 20, 'v_layers': 5, 'v_hidden_dim': 5, 'n1': 10, 'n2': 5, 'u_rate': 0.0002133670132946601, 'v_rate': 0.0006349874886164373}
# Best trial config: {'u_layers': 2, 'u_hidden_dim': 20, 'v_layers': 4, 'v_hidden_dim': 20, 'n1': 10, 'n2': 7, 'u_rate': 0.0006756995220370141, 'v_rate': 0.0002804879090959305}
# Best trial config: {'u_layers': 4, 'u_hidden_dim': 23, 'v_layers': 4, 'v_hidden_dim': 15, 'n1': 10, 'n2': 6, 'u_rate': 0.003370553415547106, 'v_rate': 0.009087847200586583}
# Best trial config: {'u_layers': 7, 'u_hidden_dim': 30, 'v_layers': 7, 'v_hidden_dim': 50, 'n1': 10, 'n2': 6, 'u_rate': 0.07933644599535282, 'v_rate': 0.03159263256623099}

ray.init(num_cpus=4, num_gpus=3)
grace_period=config['iteration']// config['subiteration']+1
analysis = tune.run(
    train,
    num_samples=150,
    scheduler=ASHAScheduler(metric="loss", mode="min",  grace_period=grace_period, max_t=20*grace_period, reduction_factor=4),#
    config=config,
    verbose=2,
#     local_checkpoint_dir='/home/wuy/ray_results/train_2021-01-30_23-34-27/',
    local_dir='/home/wuy/ray_results/train_2021-01-30_23-34-27/',
    #/train_2021-01-30_23-34-27/train_b1a42_00143_143_alpha\=1903.1\,u_rate\=0.0029375\,v_rate\=0.0023904_2021-02-02_21-40-56/checkpoint_7780
    resources_per_trial = {"gpu": 1},
#     resume=True,
    resume='LOCAL'
#     num_gpus=1
)

### max_t>=grace_period

best_trial = analysis.get_best_trial(metric="loss", mode="min")
print("Best trial config: {}".format(best_trial.config))



best_trained_generator = generator(best_trial.config).to(device)
best_trained_discriminator = discriminator(best_trial.config).to(device)




best_checkpoint_dir =best_trial.checkpoint.value

print(os.path.join(best_checkpoint_dir, "checkpoint"))
generator_state, discriminator_state = torch.load(os.path.join(
    best_checkpoint_dir, "checkpoint"))### The purpose?


# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
Root_dir='/scratch/wuy/wan-master/newNCDE/results/'
# If we wanted to save our dataframe, appending the time so that they do not overwrite each other
# (if you have multiple sets of code running over different clusters)
df.to_csv(Root_dir+"results_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt")

'''
# Obtain a trial dataframe from all run trials of this `tune.run` call.
dfs = analysis.trial_dataframes


ax = None  # This plots everything on the same plot
for d in dfs.values():
    ax = d.mean_error.plot(ax=ax, legend=False)
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean Error")


x_mesh = torch.linspace(down, up, 500, requires_grad=True)
mesh1, mesh2 = torch.meshgrid(x_mesh, x_mesh)
mesh_1 = torch.reshape(mesh1, [-1, 1]).repeat(1, 11).unsqueeze(2).view(-1, 1, 11)
mesh_2 = torch.reshape(mesh2, [-1, 1]).repeat(1, 11).unsqueeze(2).view(-1, 1, 11)
t = torch.linspace(0, 1, 11).unsqueeze(1).view(1, -1).repeat(250000, 1).unsqueeze(2).view(-1, 1, 11)
xt_comparison = torch.cat((mesh_1, mesh_2, t), dim=1)

u_net = generator(config)

error = torch.sqrt(torch.square(func_u_sol(xt_comparison)-u_net(xt_comparison[:, 0, :], xt_comparison[:, 1, :], xt_comparison[:, 2, :]).squeeze(2)))
# mean_error = torch.mean(error, dim=1)
# mean_error = mean_error.view(500, 500)
l_error = error[:, -1].view(500,500)

plt.figure(figsize=(10, 6))

cset=plt.contourf(mesh1.data.numpy(),mesh2.data.numpy(), l_error.data.numpy(), 500, cmap='winter')

plt.colorbar(cset)
plt.show()
'''

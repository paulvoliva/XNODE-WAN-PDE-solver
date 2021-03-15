from data.dataset import Path, FixedCDEDataset, FlexibleCDEDataset, SubsampleDataset
from data.scalers import TrickScaler
from data.intervals import FixedIntervalSampler, RandomSampler, BatchIntervalSampler, create_interval_dataloader
from data.functions import torch_ffill
from rdeint import rdeint
from model import NeuralRDE
import torch
import math

'''Given a data tensor of shape [N, L, C] that is filled with nan values, and a corresponding times tensor of shape
    [N, L] the corresponds to the time the data was collected for each row
    
    The data of shape [N, L, C]. It is assumed that the times are in the first index of the
    data channels.
    
    N = number of paths
    L = time dim
    C = channels'''

#'''
# I use this to get the shape of the tensor when I debug
old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info
#'''

''' # Parameters'''

npaths = 20
t_mesh_size = 100
step = 4
batch_size = npaths
depth = 3

assert npaths==batch_size, "Warning! The code does not yet support actual batches of data, modify the training function to do so"

bnpaths = 20 #number of paths on the boundary (will be multiplied by 4)

up = 1
down = -1
T = 100             #remember to divide this by 100 in all of the functions
T0 = 0

''' # Dataset '''

x = torch.Tensor(npaths, t_mesh_size, 1).uniform_(down, up).requires_grad_(True)
y = torch.Tensor(npaths, t_mesh_size, 1).uniform_(down, up).requires_grad_(True)
t = torch.linspace(T0, T, t_mesh_size).unsqueeze(1).unsqueeze(2).view(1, t_mesh_size, 1).repeat(npaths, 1, 1).requires_grad_(True)

xv = x[:, ::step, :].clone().detach().requires_grad_(True)
yv = y[:, ::step, :].clone().detach().requires_grad_(True)
tv = t[:, ::step, :].clone().detach().requires_grad_(True)

txy = torch.cat((t, x, y), dim=2)

dataset = FlexibleCDEDataset(txy, torch.ones_like(txy)[:, :, 0], depth=depth)
sampler = FixedIntervalSampler(t_mesh_size, step, from_start=True, include_end=False)

dataloader = create_interval_dataloader(dataset, sampler, batch_size)

# boundary points

r1, r2, r3, r4 = torch.Tensor(bnpaths, t_mesh_size, 1).uniform_(down, up), torch.Tensor(bnpaths, t_mesh_size, 1).uniform_(down, up), torch.Tensor(bnpaths, t_mesh_size, 1).uniform_(down, up), torch.Tensor(bnpaths, t_mesh_size, 1).uniform_(down, up)
ones = torch.ones(bnpaths, t_mesh_size, 1)
t1, t2, t3, t4 = torch.Tensor(bnpaths, t_mesh_size, 1).uniform_(T0, T), torch.Tensor(bnpaths, t_mesh_size, 1).uniform_(T0, T), torch.Tensor(bnpaths, t_mesh_size, 1).uniform_(T0, T), torch.Tensor(bnpaths, t_mesh_size, 1).uniform_(T0, T)

bl, br, bt, bb = torch.cat((t1, -ones, r1), dim=2), torch.cat((t2, ones, r2), dim=2), torch.cat((t3, r3, ones), dim=2), torch.cat((t4, r4, -ones), dim=2)
btxy = torch.cat((bl, br, bb, bt), dim=0)

borderset = FlexibleCDEDataset(btxy, torch.ones_like(btxy)[:, :, 0], depth=depth)
borderloader = create_interval_dataloader(borderset, sampler, 4*bnpaths)

''' # PDE functions '''

def func_u_sol(x, y, t):
    u = 2 * torch.sin(math.pi / 2 * x) * torch.cos(math.pi / 2 * y) * torch.exp(t/100)
    return(u)

def func_f(x, y, t):
    f = (math.pi ** 2 - 2) * torch.sin(math.pi / 2 * x) * torch.cos(math.pi / 2 * y) * torch.exp(
        -t/100) - 4 * torch.sin(math.pi / 2 * x) ** 2 * torch.cos(math.pi / 2 * y) * torch.exp(-t/100)
    return(f)

def func_g(bx, by, bt):
    # bx, by, bt denote the boundary coordinates
    return func_u_sol(bx, by, bt)

def func_h(x, y):
    h = 2 * torch.sin(math.pi / 2 * x) * torch.cos(math.pi / 2 * y)
    return h

def func_w(x):  # returns 1 for positions in the domain and 0 otherwise
    w_bool = torch.gt(1 - torch.abs(x), torch.zeros(x.shape)) & torch.gt(torch.abs(x), torch.zeros(x.shape))
    w_val = torch.where(w_bool, 1 - torch.abs(x) + torch.abs(x), torch.zeros(x.shape))
    return w_val

''' # Model'''

config = {
    'v_layers': 5,
    'v_hidden_dim': 10,
    'u_hidden_dim': 10,
    'u_hidden_hidden_dim': 15,
    'u_layers': 5,
    'alpha': 25,  # 1e5*boundary_sample_size*4
    'n1': 2,
    'n2': 1,
    'u_rate': 0.0015,
    'v_rate': 0.0015
}

# The generator model is just the NeuralRDE model

class discriminator(torch.nn.Module):  # this makes the v function
    def __init__(self, config):
        super().__init__()
        self.num_layers = config['v_layers']
        self.hidden_dim = config['v_hidden_dim']
        self.input = torch.nn.Linear(3, self.hidden_dim)
        self.hidden = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = torch.nn.Linear(self.hidden_dim, 1)
        self.net = torch.nn.Sequential(*[
            self.input,
            *[torch.nn.ReLU(), self.hidden] * self.num_layers,
            torch.nn.Tanh(),
            self.output

        ])

    def forward(self, x, y, t):
        inp = torch.cat((t, x, y), dim=2)
        x = self.net(inp)
        return x

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return (self.loss)

''' # Loss '''

def I(y_output_u, y_output_v, xv, yv, tv, x, y, t):
    y_output_u.retain_grad()
    y_output_v.retain_grad()
    phi = y_output_v * func_w(xv)
    y_output_u.backward(torch.ones_like(y_output_u), retain_graph=True)
    du_x = x.grad[:, ::step, :]
    du_y = y.grad[:, ::step, :]
    phi.backward(torch.ones_like(phi), retain_graph=True)
    dphi_x = xv.grad
    dphi_y = yv.grad
    dphi_t = tv.grad
    s1 = y_output_u[:, -1, :] * phi[:, -1, :] - func_h(x[:, 0, :], y[:, 0, :]) * phi[:, 0, :]
    s2 = (y_output_u * dphi_t)/t_mesh_size*step  # for t does this make sense?
    Lap = du_x * dphi_x + du_y * dphi_y
    s3 = (T-T0)*(Lap + y_output_u * y_output_u * phi - func_f(x[:, ::step, :], y[:, ::step, :], t[:, ::step, :]) * phi)/t_mesh_size*step
    I = torch.sum(s1 - torch.sum(s2 - s3, 1), 0)
    x.grad.data.zero_()
    y.grad.data.zero_()
    xv.grad.data.zero_()
    yv.grad.data.zero_()
    tv.grad.data.zero_()
    return I


def L_init(y_output_u):
    return torch.mean((y_output_u[:, 0, :] - func_h(x[:, 0, :], y[:, 0, :]) ** 2))


def L_bdry(u_net, border_logsig):
    return torch.mean((u_net(border_logsig) - func_g(btxy[:, ::step, 1], btxy[:, ::step, 2], btxy[:, ::step, 0]).unsqueeze(2)) ** 2)


def L_int(y_output_u, y_output_v, xv=xv, yv=yv, tv=tv, x=x, y=y, t=t):
    # x needs to be the set of points set plugged into net_u and net_v
    return torch.log((I(y_output_u, y_output_v, xv, yv, tv, x, y, t)) ** 2) - torch.log(torch.sum(y_output_v ** 2))


def Loss_u(y_output_u, y_output_v, border_logsig, u_net, alpha, gamma):
    return L_int(y_output_u, y_output_v) + gamma * L_init(y_output_u) + alpha * L_bdry(
        u_net, border_logsig)

def Loss_v(y_output_u, y_output_v):
    return -L_int(y_output_u, y_output_v)

iteration = 100

(initial, logsig), responses = next(iter(dataloader))
logsig_dim = logsig.shape[2]

def train(config):
    n1 = config['n1']
    n2 = config['n2']

    # neural network models
    u_net = NeuralRDE(3, logsig_dim, config['u_hidden_dim'], 1, hidden_hidden_dim=config['u_hidden_hidden_dim'], num_layers=config['u_layers'], return_sequences=True)
    v_net = discriminator(config)

    # optimizers for WAN
    optimizer_u = torch.optim.Adam(u_net.parameters(), lr=config['u_rate'])
    optimizer_v = torch.optim.Adam(v_net.parameters(), lr=config['v_rate'])

    batch, reponses = next(iter(dataloader))
    prediction_u = u_net(batch)
    prediction_v = v_net(xv, yv, tv)

    for k in range(iteration):

        for i in range(n1):
            batch, reponses = next(iter(dataloader))
            batchborder, responses = next(iter(borderloader))
            loss_u = Loss_u(prediction_u, prediction_v, batchborder, u_net, config['alpha'], config['alpha'])
            optimizer_u.zero_grad()
            loss_u.backward(retain_graph=True)
            optimizer_u.step()
            prediction_u = u_net(batch)

        for j in range(n2):
            loss_v = Loss_v(prediction_u, prediction_v)
            optimizer_v.zero_grad()
            loss_v.backward(retain_graph=True)
            optimizer_v.step()
            prediction_v = v_net(xv, yv, tv)


        if k % 10 == 0:
            print(k, loss_u.data, loss_v.data)
            error_test = torch.mean(
                torch.sqrt(torch.square((func_u_sol(x[:, ::step, :], y[:, ::step, :], t[:, ::step, :]) - prediction_u.data)))).data
            print("error test " + str(error_test))

'''
for batch in dataloader:
    values, responses = batch
    model = NeuralRDE(3, 14, 10, 1, return_sequences=True)
    r = model(values)
    M = discriminator(config)
    R = M(xv, yv, tv)
    k = I(r, R, xv, yv, tv, x, y, t)
'''

train(config)

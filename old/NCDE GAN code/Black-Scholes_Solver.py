from data.dataset import Path, FixedCDEDataset, FlexibleCDEDataset, SubsampleDataset
from data.scalers import TrickScaler
from data.intervals import FixedIntervalSampler, RandomSampler, BatchIntervalSampler, create_interval_dataloader
from data.functions import torch_ffill
from rdeint import rdeint
from model import NeuralRDE
import torch
import math
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os
import signatory
from itertools import product

'''Given a data tensor of shape [N, L, C] that is filled with nan values, and a corresponding times tensor of shape
    [N, L] the corresponds to the time the data was collected for each row
    
    The data of shape [N, L, C]. It is assumed that the times are in the first index of the
    data channels.
    
    N = number of paths
    L = time dim
    C = channels'''

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
Here we are solving the Black-Scholes equation with $r=0$:
\begin{equation}
\left\{\begin{array}{ll}
\partial_t V + \frac{1}{2} \sigma^2 S^2 \partial^2_S V = 0 & \text{on} (0,100)\times(0,100)\\
V(0,t) = 0\\
V(100,t) = 100\\
V(S,100) = \max{S-20,0}
\end{array}\right.
\end{equation}
'''

'''
# I use this to get the shape of the tensor when I debug
old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info
#'''

def train(config, checkpoint_dir=None):

    # setting to cuda
    if torch.cuda.is_available():
      dev = "cuda:0"
    else:
      dev = "cpu"

    device = torch.device(dev)
    #print(device)

    ''' # Parameters'''

    npaths = 20
    intervals = 30
    step = 10
    t_mesh_size = int(step*intervals)
    batch_size = npaths
    depth = 1

    assert npaths==batch_size, "Warning! The code does not yet support actual batches of data, modify the training function to do so"

    bnpaths = 20 #number of paths on the boundary (will be multiplied by 4)

    up = 100
    down = 0
    T = 100             #remember to divide this by 100 in all of the functions
    T0 = 0
    dim = 1

    ''' # Dataset '''

    x = torch.Tensor(npaths, t_mesh_size, 1).uniform_(down, up).to(device).requires_grad_(True)
    t = torch.linspace(T0, T, t_mesh_size).unsqueeze(1).unsqueeze(2).view(1, t_mesh_size, 1).repeat(npaths, 1, 1).to(device).requires_grad_(True)

    X = [x, t]

    xv = x[:, ::step, :].clone().detach().to(device).requires_grad_(True)
    tv = t[:, ::step, :].clone().detach().to(device).requires_grad_(True)

    XV = [xv, tv]

    tx = torch.cat((t, x), dim=2).to(device)

    dataset = FlexibleCDEDataset(tx, torch.ones_like(tx)[:, :, 0], depth=depth)
    sampler = FixedIntervalSampler(t_mesh_size, step, from_start=True, include_end=False)

    dataloader = create_interval_dataloader(dataset, sampler, batch_size)

    # boundary points

    ones = torch.ones(bnpaths, t_mesh_size, 1)
    t1, t2 = torch.Tensor(bnpaths, t_mesh_size, 1).uniform_(T0, T), torch.Tensor(bnpaths, t_mesh_size, 1).uniform_(T0, T)

    bl, br = torch.cat((t1, down*ones), dim=2), torch.cat((t2, up*ones), dim=2)
    btx = torch.cat((bl, br), dim=0)

    borderset = FlexibleCDEDataset(btx, torch.ones_like(btx)[:, :, 0], depth=depth)
    borderloader = create_interval_dataloader(borderset, sampler, 2*bnpaths)

    ''' # PDE functions '''

    def func_f(x, t):
        return(torch.zeros(1,1))

    def func_g(bx, bt):
        # bx, by, bt denote the boundary coordinates
        return bx

    def func_h(x):
        h,i = torch.max(torch.cat(((x-20).unsqueeze(2), torch.zeros(x.size()).unsqueeze(2).to(device)), dim=2), dim=2)
        return h

    def func_c(y_output_u):
        return 0

    # this is meant to be a d by d-dimensional array containing npaths by intervals by 1 tensors
    a = torch.ones(1, 1, npaths, intervals, 1).to(device)

    # this is meant to be a d-dimensional containing npaths by intervals by 1 tensors
    b = torch.zeros(1, npaths, intervals, 1).to(device)

    x_setup = xv.clone().detach().to(device)
    t_setup = tv.clone().detach().to(device)
    btx_setup = btx[:, ::step, :].clone().detach()

    h = func_h(x_setup[:, -1, :]).to(device)
    f = func_f(x_setup, t_setup).to(device)
    g = func_g(btx_setup[:, :, 1], btx_setup[:, :, 0]).unsqueeze(2).to(device)

    def func_w(x):  # returns 1 for positions in the domain and 0 otherwise
        w_bool = torch.gt(1 - torch.abs(x), torch.zeros(x.shape).to(device)) & torch.gt(torch.abs(x), torch.zeros(x.shape).to(device))
        w_val = torch.where(w_bool, 1 - torch.abs(x) + torch.abs(x), torch.zeros(x.shape).to(device))
        return w_val

    ''' # Model'''

    # The generator model is just the NeuralRDE model

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

        def forward(self, inp):
            x = self.net(inp)
            return x

        def backward(self, retain_graph=True):
            self.loss.backward(retain_graph=retain_graph)
            return (self.loss)

    ''' # Loss '''

    def I(y_output_u, y_output_v, XV, X, a=a, b=b,h=h, f=f, c=func_c):
        y_output_u.retain_grad()
        y_output_v.retain_grad()
        phi = y_output_v * func_w(xv)
        y_output_u.backward(torch.ones_like(y_output_u), retain_graph=True)
        du = {}
        for i in range(dim):
            du['du_'+str(i)] = X[i].grad[:, ::step, :]
        phi.backward(torch.ones_like(phi), retain_graph=True)
        dphi = {}
        for i in range(dim+1):
            dphi['dphi_'+str(i)] = XV[i].grad
        s1 = y_output_u[:, -1, :] * phi[:, -1, :] - h
        s2 = (y_output_u * dphi['dphi_1'])/t_mesh_size*step  # for t does this make sense?
        s31 = 0
        for i,j in product(range(dim), repeat=2):
            s31 += a[i, j, :, :, :] * dphi['dphi_'+str(i)] * du['du_'+str(j)]
        s32 = 0
        for i in range(dim):
            s32 += b[i] * phi * du['du_'+str(i)]
        s3 = (T-T0)*(s31 + s32 + func_c(y_output_u) * y_output_u * phi - f * phi)/t_mesh_size*step
        I = torch.sum(s1 - torch.sum(s2 - s3, 1), 0)
        for i in X:
            i.grad.data.zero_()
        for i in XV:
            i.grad.data.zero_()
        return I


    def L_init(y_output_u, h=h):
        return torch.mean((y_output_u[:, -1, :] - h )** 2)


    def L_bdry(u_net, border_logsig, g=g):
        return torch.mean((u_net(border_logsig) - g) ** 2)


    def L_int(y_output_u, y_output_v, XV=XV, X=X):
        # x needs to be the set of points set plugged into net_u and net_v
        return torch.log((I(y_output_u, y_output_v, XV, X)) ** 2) - torch.log(torch.sum(y_output_v ** 2))


    def Loss_u(y_output_u, y_output_v, border_logsig, u_net, alpha, gamma):
        return L_int(y_output_u, y_output_v) + gamma * L_init(y_output_u) + alpha * L_bdry(
            u_net, border_logsig)

    def Loss_v(y_output_u, y_output_v):
        return -L_int(y_output_u, y_output_v)

    iteration = 2000

    (initial, logsig), responses = next(iter(dataloader))
    logsig_dim = logsig.shape[2]

    '''
    batch, reponses = next(iter(dataloader))
    batchborder, responses = next(iter(borderloader))
    '''

    n1 = config['n1']
    n2 = config['n2']

    # neural network models
    u_net = NeuralRDE(dim+1, logsig_dim, config['u_hidden_dim'], 1, hidden_hidden_dim=config['u_hidden_hidden_dim'], num_layers=config['u_layers'], return_sequences=True).to(device)
    v_net = discriminator(config).to(device)

    # optimizers for WAN
    optimizer_u = torch.optim.Adam(u_net.parameters(), lr=config['u_rate'])
    optimizer_v = torch.optim.Adam(v_net.parameters(), lr=config['v_rate'])

    v_inp = XV[-1]
    for i in range(dim):
        v_inp = torch.cat((v_inp, XV[i]), dim=2)

    for batch, responses in dataloader:
        prediction_u = u_net(batch)
    prediction_v = v_net(v_inp)

    Loss = 0

    for k in range(iteration):

        for i in range(n1):
            for batch, responses in dataloader:
                for batchborder, responses in borderloader:
                    (initial, logsig), (binitial, blogsig) = batch, batchborder
                    batch, batchborder = (initial.to(device), logsig.to(device)), (binitial.to(device), blogsig.to(device))
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
            prediction_v = v_net(v_inp)

        Loss += 0.1*loss_u

        if k % 10 == 0:
            print(k, loss_u.data, loss_v.data)
            #error_test = torch.mean(
                #torch.sqrt(torch.square((func_u_sol(x[:, ::step, :], y[:, ::step, :], t[:, ::step, :]) - prediction_u.data)))).data
            #print("error test " + str(error_test))

            tune.report(Loss=Loss.item())

            with tune.checkpoint_dir(k) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((u_net.state_dict(), v_net.state_dict()), path)
            '''
config = {
    'v_layers': 5,
    'v_hidden_dim': 10,
    'u_hidden_dim': 15,
    'u_hidden_hidden_dim': 5,
    'u_layers': 5,
    'alpha': 25,  # 1e5*boundary_sample_size*4 # 25
    'n1': 2,
    'n2': 1,
    'u_rate': 0.0015,       # 0.0015
    'v_rate': 0.0015       # 0.0015
}
'''
config = {
    'v_layers': tune.qrandint(2, 10),   # 5
    'v_hidden_dim': tune.qrandint(5, 20),  # 10
    'u_hidden_dim': tune.qrandint(5, 20),
    'u_hidden_hidden_dim': tune.qrandint(5, 20),
    'u_layers': tune.qrandint(5, 20),
    'alpha': tune.loguniform(10, 1e7),  # 1e5*boundary_sample_size*4 # 25
    'n1': 2,
    'n2': 1,
    'u_rate': tune.loguniform(1e-6, 1e-2),       # 0.0015
    'v_rate': tune.loguniform(1e-6, 1e-2)       # 0.0015
}
'''

train(config)

'''
tune.utils.diagnose_serialization(train)

analysis = tune.run(
    train,
    num_samples=200,
    scheduler=ASHAScheduler(metric="Loss", mode="min", grace_period=10, max_t=200, reduction_factor=4),
    config=config,
    verbose=2,
    resources_per_trial={'cpu':1}
)

best_trial = analysis.get_best_trial(metric="Loss", mode="min")
print("Best trial config: {}".format(best_trial.config))
'''

best_trained_generator = NeuralRDE(3, logsig_dim, best_trial.config['u_hidden_dim'], 1, hidden_hidden_dim=best_trial.config['u_hidden_hidden_dim'], num_layers=best_trial.config['u_layers'], return_sequences=True)
best_trained_discriminator = discriminator(best_trial.config)

best_checkpoint_dir = best_trial.checkpoint.value
generator_state, discriminator_state = torch.load(os.path.join(
    best_checkpoint_dir, "checkpoint"))
'''

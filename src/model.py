"""
model.py
===========================
This contains a model class for NeuralRDEs that wraps `rdeint` as a `nn.Module`.
"""
import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint
from src.dataset import fillt


def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)


class discriminator(torch.nn.Module):
    '''
    This function is the discriminator and will be the function that will give us the test function. This model can
    intake an arbitrarily long list of inputs but all the lists need to be equally long. The input shape is [L, C, T]
    where L is the number of points, C is the number of dimensions and T is the number of
    time points.

    config: dictionary containing all the hyperparameters ('v_layers' and 'v_hidden_dim' for the
                    discriminator)
    setup: dictionary containing information of the problem
    '''

    def __init__(self, config: dict, setup: dict):
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
        self.net.double()

    def forward(self, XV: torch.Tensor):
        x = self.net(XV.double())
        return x

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return (self.loss)


class NeuralODE(nn.Module):
    '''
    The generic module for learning with Neural ODEs.

    This class wraps the `_F` field that acts like an RNN-Cell. This method simply initialises the hidden dynamics
    and computes the updated hidden dynamics through a call to `ode_int` using the `_F` as the function that
    computes the update.
    '''
    def __init__(self, hidden_dim: int, output_dim: int, func_h, func_g, setup: dict, hidden_hidden_dim: int, num_layers: int,
                 domain, solver: str = 'midpoint', min_steps: int = 5, adjoint: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.h = func_h
        self.g = func_g     # initial conditions
        self.setup = setup
        self.hidden_hidden_dim = hidden_hidden_dim
        self.num_layers = num_layers
        self.domain = domain
        self.solver = solver
        self.min_steps = min_steps
        self.adjoint = adjoint

        # Initial to hidden
        self.initial_layers = nn.Sequential(*[nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]).double()

        # The net applied to h_prev
        self.ODE_rhs = _ODEField(hidden_dim, setup, hidden_dim=hidden_hidden_dim, num_layers=num_layers)
        self.ODE_rhs.apply(init_weights)

        # Linear classifier to apply to final layer
        self.final_linear = nn.Linear(self.hidden_dim, self.output_dim).double()

    def forward(self, inputs: torch.Tensor):
        # Setup the inital hidden layer
        if inputs.shape[1] == 1 and inputs[0, 0, 0] == self.setup['T0']:
            h_ = self.h(inputs[:, 0, :]).unsqueeze(1).double()
            return self.final_linear(self.initial_layers(h_))
        path_i, i, timesteps = (None, None, inputs[0, :, 0]) if \
            torch.max(self.domain.func_w(inputs[:, 0, :].unsqueeze(1))) < 1e-5 or inputs[0, 0, 0] == self.setup['T0'] \
            else self.domain.bound_pad(inputs)  # filling the timesteps if they are not enough and do not reach the boundary
        h_ = self.h(inputs[:, 0, :]).unsqueeze(1).double() if inputs[0, 0, 0] == self.setup['T0'] else \
            self.g(inputs[:, 0, :].unsqueeze(1)).double()  # initial value
        h0 = self.initial_layers(h_)    # lifting to the higher dimensional space

        field = _F(self.ODE_rhs, inputs[:, 0, 1:])


        # Solve
        odeint_func = odeint_adjoint if self.adjoint else odeint
        out = odeint_func(func=field, y0=h0, t=timesteps, method=self.solver).transpose(0, 1) if i is None \
            else ([odeint_func(func=field, y0=h0, t=times.squeeze(), method=self.solver).transpose(0, 1)[pi][:, idx] for times, idx, pi in zip(timesteps, i, path_i)]\
            if path_i is not None else odeint_func(func=field, y0=h0, t=timesteps, method=self.solver).transpose(0, 1)[:, i])

        # Outputs
        out = torch.cat(tuple(out), dim=0) if isinstance(out, list) else out
        out_ = self.final_linear(out)   # reduce to output dimensions

        return out_


class _ODEField(nn.Module):
    '''
    The hidden field, modelled as a DNN, over which we solve the ODE. This is the field F s.t.

    dh/dt = F

    where h is our hidden state.
    '''
    def __init__(self, input_dim: int, setup: dict, num_layers: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Additional layers are just hidden to hidden with relu activation
        additional_layers = [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)] * (num_layers - 1) if num_layers > 1 else []

        # The net applied to h_prev
        self.net = nn.Sequential(*[
            nn.Linear(input_dim+setup['dim']+1, hidden_dim),
            *additional_layers,
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        ]).double() if num_layers > 0 else nn.Linear(input_dim, input_dim-1).double()

    def forward(self, h: torch.Tensor):
        return self.net(h.squeeze())


class _F(nn.Module):
    '''
    In this class, we include (t,x,h) as the input to the DNN network of the vector field.
    '''
    def __init__(self, func, x: torch.Tensor):
        super().__init__()
        self.func = func
        self.x = x

    def forward(self, t: torch.Tensor, h: torch.Tensor):
        x_ = torch.cat((self.x, t.repeat(h.shape[0], 1), h), dim=1)
        A = self.func(x_)
        return A

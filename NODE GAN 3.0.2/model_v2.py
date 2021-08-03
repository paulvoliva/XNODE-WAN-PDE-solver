"""
model_v2.py
===========================
This contains a model class for NeuralRDEs that wraps `odeint_` as a `nn.Module`.
"""
import torch
from torch import nn
from odeint_ import odeint_

def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)

class NeuralODE(nn.Module):
    """The generic module for learning with Neural RDEs.

    This class wraps the `NeuralODECell` that acts like an RNN-Cell. This method simply initialises the hidden dynamics
    and computes the updated hidden dynamics through a call to `ode_int` using the `NeuralODECell` as the function that
    computes the update.

    Here we model the dynamics of some abstract hidden state H via a CDE, and the response as a linear functional of the
    hidden state, that is:
        dH = f(H)dX;    Y = L(H).
    """
    def __init__(self,
                 logsig_dim,
                 hidden_dim,
                 output_dim,
                 func_h,
                 setup,
                 hidden_hidden_dim=15,
                 num_layers=3,
                 apply_final_linear=True,
                 solver='midpoint',
                 adjoint=False,
                 return_sequences=False):
        """
        Args:
            logsig_dim (int): The dimension of the log-signature. ###delete?
            hidden_dim (int): The dimension of the hidden state.
            output_dim (int): The dimension of the output.
            hidden_hidden_dim (int): The dimension of the hidden layer in the RNN-like block.
            num_layers (int): The number of hidden layers in the vector field. Set to 0 for a linear vector field.
            apply_final_linear (bool): Set False to ignore the final linear output.
            solver (str): ODE solver, must be implemented in torchdiffeq.
            adjoint (bool): Set True to use odeint_adjoint.
            return_sequences (bool): If True will return the linear function on the final layer, else linear function on
                all layers.
        """
        super().__init__()
        self.logsig_dim = logsig_dim#???need delete?
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.h = func_h
        self.hidden_hidden_dim = hidden_hidden_dim
        self.num_layers = num_layers
        self.apply_final_linear = apply_final_linear
        self.solver = solver
        self.adjoint = adjoint
        self.return_sequences = return_sequences

        # Initial to hidden
        self.initial_layers = nn.Sequential(*[nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]).double()

        # The net applied to h_prev
        self.func = _NODEFunc(hidden_dim, logsig_dim, setup, hidden_dim=hidden_hidden_dim, num_layers=num_layers)
        self.func.apply(init_weights)

        # Linear classifier to apply to final layer
        self.final_linear = nn.Linear(self.hidden_dim, self.output_dim).double() if apply_final_linear else lambda x: x

    def forward(self, vals):
        # Setup the initial hidden layer
        timesteps = vals[0, :, 0] if vals.shape[1] != 1 else (vals[0, :, 0] if vals[0, :, 0].item()==0 else torch.linspace(0, vals[0, :, 0].item(), 50).to(vals.device))
        h_ = self.h(vals[:, 0, :]).unsqueeze(1)
        h0 = self.initial_layers(h_)

        if vals.shape[1] == 1:
            self.return_sequences = False
        else:
            self.return_sequences = True

        # Perform the adjoint operation
        out = odeint_(
            timesteps, h0, self.func, vals[:, 0, :][:, 1:], method=self.solver, adjoint=self.adjoint, return_sequences=self.return_sequences
        )

        # Outputs
        outputs = self.final_linear(out[:, -1, :]).unsqueeze(2) if not self.return_sequences else self.final_linear(out)

        return outputs


class _NODEFunc(nn.Module):
    """The function applied to the hidden state in the log-ode method.

    This creates a simple RNN-like block to be used as the computation function f in:
        dh/dt = f(h) o X_{[t_i, t_{i+1}]}???

    To build a custom version, simply use any NN architecture such that `input_dim` is the size of the hidden state,
    and the output dim must be of size `input_dim * ??_dim`. Simply reshape the output onto a tensor of size
    `[batch, input_dim, ??]`.
    """
    def __init__(self, input_dim, logsig_dim, setup, num_layers=1, hidden_dim=15):
        super().__init__()
        self.input_dim = input_dim
        self.logsig_dim = logsig_dim ###do we need it?
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

    def forward(self, h):
        return self.net(h)


if __name__ == '__main__':
    NeuralODE(10, 20, 15, 5, hidden_hidden_dim=90, num_layers=3)


"""
odeint_.py
===========================
Contains the rde-equivalent of the torchdiffeq `odeint` and `odeint_adjoint` functions.
"""
import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint
import bisect


def odeint_(timesteps, h0, func, x, method='rk4', adjoint=False, return_sequences=False):
    """Analogous to odeint but for ODEs.

    Note that we do not have time intervals here. This is because the log-ode method is always evaluated on [0, 1] and
    thus are grid is always [0, 1, ..., num_intervals+1].

    Args:
        logsig (torch.Tensor): A tensor of logsignature of shape [N, L, logsig_dim]
        h0 (torch.Tensor): The initial value of the hidden state.
        func (nn.Module): The function to apply to the state h0.
        method (str): The solver to use.
        adjoint (bool): Set True to use the adjoint method.
        return_sequences (bool): Set True to return a prediction at each step, else return just terminal time.

    Returns:
        torch.Tensor: The values of the hidden states at the specified times. This has shape [N, L, num_hidden].
    """

    # A cell to apply the output of the function linearly to correct log-signature piece.
    cell = _NODECell(func, x)

    # Solve
    odeint_func = odeint_adjoint if adjoint else odeint
    output = odeint_func(func=cell, y0=h0, t=timesteps, method=method).transpose(0, 1)

    return output


def set_options(timesteps, device, return_sequences=False, eps=1e-5):
    """Sets the options to be passed to the relevant `odeint` function.

    Args:
        logsig (torch.Tensor): The logsignature of the path.
        return_sequences (bool): Set True if a regression problem where we need the full sequence. This requires us
            specifying the time grid as `torch.arange(0, T_final)` which is less memory efficient that specifying
            the times `t = torch.Tensor([0, T_final])` along with an `step_size=1` in the options.
        eps (float): The epsilon perturbation to make to integration points to distinguish the ends.

    Returns:
        torch.Tensor, dict: The integration times and the options dictionary.
    """
    length = timesteps
    if return_sequences:
        t = torch.arange(0, length, dtype=torch.float).to(device)
        options = {'eps': eps}
    else:
        options = {'step_size': 1, 'eps': eps}
        t = torch.Tensor([0, length]).to(device)
    return t, options


class _NODECell(nn.Module):
    """Applies the function to the previous hidden state, and then applies the output linearly onto the spatial variable.

    The NeuralRDE model solves the following equation:
        dH = f(H) o X_{t_i, t_{i+1} dt;    H(0) = H_t_i.
    given a function f, this class applies that function to the hidden state, and then applies that result linearly onto
    the correct piece of X.
    """
    def __init__(self, func, x):
        super().__init__()
        self.func = func
        self.x = x

    def forward(self, t, h):
        x_ = torch.cat((self.x, t.repeat(h.shape[0], 1), h), dim=1)
        A = self.func(x_)
        return A



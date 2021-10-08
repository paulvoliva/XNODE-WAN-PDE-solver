import numpy as np
import torch
import matplotlib.pyplot as plt
import json


def L_norm(X: torch.Tensor, u_net, p: float, func_u_sol, volume: float, N_r: int, error=True):
    '''
    this function will compute the error of our guess in L^p norm
    '''
    diff = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not isinstance(X, list):
        f = func_u_sol(X).to(device)-u_net(X.to(device)).squeeze() if error else func_u_sol(X).to(device)
        return (volume * torch.mean(torch.pow(torch.abs(f), p))) **(1/p)
    for x in X:
        rel = x.shape[0]/N_r
        f = func_u_sol(x).to(device) - u_net(x.to(device)).squeeze() if error else func_u_sol(x).to(device)
        diff += rel * torch.mean(torch.pow(torch.abs(f), p))

    norm = (volume * diff) ** (1 / p)
    return norm


def rel_err(X: torch.Tensor, predu: torch.Tensor, func_u_sol, p: float, volume: float, N_r: int):
    '''
    this function computes the relative error in L^p norm
    '''
    rel = L_norm(X, predu, p, func_u_sol, volume, N_r)/L_norm(X, predu, p, func_u_sol, volume, N_r, error=False)
    return rel


# TODO: generalise the function
def proj(u_net, setup, iteration, device, axes=[0, 1], T=1, T0=0, save=False, show=True, resolution=100,
         colours=8, func_u_sol=0):
    # assumes hypercube and will max all free coordinates for the plot

    assert len(axes) == 2, 'There can only be two axes in the graph to be able to display them'

    down, up = setup['shape_param'] if isinstance(setup['shape_param'], list) else (-setup['shape_param'], setup['shape_param'])

    assert isinstance(down, (float, int)) and isinstance(up, (float, int)), 'The model assumes hypercube or hypersphere, you need to modify proj to match your domain shape'

    print('WARNING: we are plotting a hypercube that envelops your shape')

    xt = torch.Tensor(resolution, resolution, setup['dim'] + 1).to(device)

    for i in list(set(range(setup['dim'] + 1)) - set(axes)):
        xt[:, :, i] = 0.5 * torch.ones(resolution, resolution)

    if 0 in axes:
        t_mesh = torch.linspace(T0, T, resolution)
    else:
        t_mesh = torch.linspace(down, up, resolution)
        xt[:, :, 0] = T * torch.ones(resolution, resolution)

    x_mesh = torch.linspace(down, up, resolution)
    mesh1, mesh2 = torch.meshgrid(x_mesh, t_mesh)
    xt[:, :, axes[0]] = mesh2
    xt[:, :, axes[1]] = mesh1

    predu = u_net(xt).to(device).detach()

    plt.clf()

    if func_u_sol != 0:
        u_sol = func_u_sol(xt).to(device)
        error = predu - u_sol.unsqueeze(2)


        data = np.asarray(predu.view(resolution, resolution).cpu().numpy())
        np.save('guess_cn.npy', data)

        data = np.asarray(error.view(resolution, resolution).cpu().numpy())
        np.save('error_cn.npy', data)
        print('Saved Error')

        fig, ax = plt.subplots(3)
        aset = ax[0].contourf(x_mesh.numpy(), t_mesh.numpy(), func_u_sol(xt).view(resolution, resolution).cpu().numpy(), colours)
        bset = ax[1].contourf(x_mesh.numpy(), t_mesh.numpy(), predu.view(resolution, resolution).cpu().numpy(), colours)
        cset = ax[2].contourf(x_mesh.numpy(), t_mesh.numpy(), error.view(resolution, resolution).cpu().numpy(), colours)
        fig.colorbar(aset, ax=ax[0])
        fig.colorbar(bset, ax=ax[1])
        fig.colorbar(cset, ax=ax[2])
        ax[0].set_title('Correct Solution, Guess and Error')
    else:
        fig, ax = plt.subplots(1)
        aset = ax[0].contourf(x_mesh.numpy(), t_mesh.numpy(), predu.view(resolution, resolution).cpu().numpy(), colours)
        fig.colorbar(aset, ax=ax[0])
        ax[0].set_title('Guess Solution')

    if save:
        plt.savefig('plot_at_' + str(iteration) + '_along_' + str(axes) + '.png')
        print('Saved')

    if show:
        plt.show()
        print('Displayed')

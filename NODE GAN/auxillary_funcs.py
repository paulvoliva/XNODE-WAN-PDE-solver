import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def L_norm(X: torch.Tensor, predu: torch.Tensor, p: float, func_u_sol, volume: float):
    '''
    this function will compute the error of our guess in L^p norm
    '''
    u_sol = func_u_sol(X).to(device).unsqueeze(2)
    return (volume * torch.mean(torch.pow(torch.abs(u_sol - predu), p))) ** (1 / p)


def rel_err(X: torch.Tensor, predu: torch.Tensor, func_u_sol, p: float):
    '''
    this function computes the relative error in L^p norm
    '''
    u_sol = func_u_sol(X).to(device).unsqueeze(2)
    rel = torch.div(torch.mean(torch.abs(u_sol - predu)**p), torch.mean(torch.abs(u_sol)**p)) ** (1/p)
    return rel


# TODO: test axes without 0
def proj(u_net, setup, iteration, axes=[0, 1], down=-1, up=1, T=1, T0=0, save=False, show=True, resolution=100,
         colours=8, func_u_sol=0):
    # assumes hypercube and will max all free coordinates for the plot

    assert len(axes) == 2, 'There can only be two axes in the graph to be able to display them'

    xt = torch.Tensor(resolution, resolution, setup['dim'] + 1).to(device)

    for i in list(set(range(setup['dim'] + 1)) - set(axes)):
        xt[:, :, i] = up * torch.ones(resolution, resolution)

    if 0 in axes:
        t_mesh = torch.linspace(T0, T, resolution)
    else:
        t_mesh = torch.linspace(down, up, resolution)
        xt[:, :, 0] = T * torch.ones(resolution, resolution)

    x_mesh = torch.linspace(down, up, resolution)
    mesh1, mesh2 = torch.meshgrid(x_mesh, t_mesh)
    xt[:, :, axes[0]] = mesh2
    xt[:, :, axes[1]] = mesh1

    predu = u_net((xt[:, 0, :], xt)).to(device).detach()

    plt.clf()

    if func_u_sol != 0:
        u_sol = func_u_sol(xt).to(device)
        error = predu - u_sol.unsqueeze(2)

        fig, ax = plt.subplots(3)
        aset = ax[0].contourf(x_mesh.numpy(), t_mesh.numpy(), u_sol.view(resolution, resolution).cpu().numpy(), colours)
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

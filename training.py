from src.loss import loss
from src.model import NeuralODE, discriminator
import itertools
import torch
from itertools import product
from utils.auxillary_funcs import proj, L_norm, rel_err
from src.dataset import *


def func_eval(X: torch.Tensor, BX: torch.Tensor, setup: dict, y_output_u: torch.Tensor, func_a, func_b, func_c, func_h,
              func_f, func_g):
    '''
    for computational efficiency the functions will be evaluated here

    Args:
        X: interior data points
        BX: border data points
        y_output_u: our guess for u
        func_: all the funcs that are the coefficients and boundary values from the PDE
    '''

    h = func_h(X[:, 0, :])
    f = func_f(X)
    g = func_g(BX)

    c = func_c(X, y_output_u)

    # this is meant to be a d by d-dimensional array containing domain_sample_size by 1 tensors
    a = torch.Tensor(setup['dim'], setup['dim'], X.shape[0], X.shape[1])

    for i, j in product(range(setup['dim']), repeat=2):
        a[i, j] = func_a(X, i, j)

    # this is meant to be a d-dimensional containing domain_sample_size by 1 tensors
    b = torch.Tensor(setup['dim'], X.shape[0], X.shape[1])

    for i in range(setup['dim']):
        b[i] = func_b(X, i)

    return h.to(X.device), f.to(X.device), g.to(X.device), a.to(X.device), b.to(X.device), c.to(X.device)


def init_weights(layer):
    if type(layer) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)

# this function ensures that the layers have the right weight initialisation


class NODE_WAN_solver:
    '''
    The WAN algorithm runs in this class

    Args:
        params: dict: a dictionary with our setup and config dicts as well as the number of iterations
        func_: all the funcs that are the coefficients and boundary values from the PDE
        domain: this is the initialised class object that will generate the points we need (more on this in dataset)
        func_u_sol: the solution to our PDE if known
        p: if fun_u_sol is given the norm difference of our guess to the true solution in L^p is computed
    '''
    def __init__(self, params: dict, func_a, func_b, func_c, func_h, func_f, func_g, device, stop=None, func_u_sol=None,
                 p: float = 1):
        self.params = params
        self.func_a = func_a
        self.func_b = func_b
        self.func_c = func_c
        self.func_h = func_h
        self.func_f = func_f
        self.func_g = func_g
        self.device = device
        self.stop = stop
        self.func_u_sol = func_u_sol
        self.p = p

        i = iter(params.items())
        self.config = dict(itertools.islice(i, 13))
        self.setup = dict(itertools.islice(i, 7))
        self.iterations = dict(itertools.islice(i, 1))['iterations']
        self.domain = eval(dict(i)['domain'])
        self.n1 = self.config['n1']
        self.n2 = self.config['n2']

        domain = self.domain(self.setup['shape_param'], self.setup['dim'], self.setup['T0'], self.setup['T'],
                             self.setup['N_t'])

        # neural network models
        # TODO: apply parallel computing
        self.u_net = NeuralODE(self.config['u_hidden_dim'], 1, self.func_h, self.func_g,
                                                     self.setup, self.config['u_hidden_hidden_dim'],
                                                     self.config['u_layers'], domain, self.config['solver'],
                                                     self.config['min_steps'], self.config['adjoint']).to(device) # torch.nn.DataParallel(
        self.v_net = torch.nn.DataParallel(discriminator(self.config, self.setup)).to(device)

        self.u_net.apply(init_weights)
        self.v_net.apply(init_weights)

        # optimizers for WAN
        self.optimizer_u = torch.optim.Adam(self.u_net.parameters(), lr=self.config['u_rate'])
        self.optimizer_v = torch.optim.Adam(self.v_net.parameters(), lr=self.config['v_rate'])

        self.best_l = float('inf')
        self.av_l = 0

    def train(self, report: bool = False, report_it: int = 10, show_plt: bool = False):
        '''
        Args:
            report: whether or not to display information on the progress of the algorithm
            report_it: after how many iterations to report
            show_plt: whether or not to show a plot
        '''
        for k in range(self.iterations):
            domain = self.domain(self.setup['shape_param'], self.setup['dim'], self.setup['T0'], self.setup['T'],
                                 self.setup['N_t'])
            points = Comb_loader(self.setup['N_r'], self.setup['N_b'], domain, self.device)

            for i in range(self.n1):
                self.av_l = 0
                self.optimizer_u.zero_grad()
                for (datau, datav, bdata) in points:
                    prediction_v = self.v_net(datav)
                    prediction_u = self.u_net(datau)
                    h, f, g, a, b, c = func_eval(datau.clone().detach(), bdata.clone().detach(), self.setup,
                                                 prediction_u, self.func_a, self.func_b, self.func_c, self.func_h,
                                                 self.func_f, self.func_g)
                    Loss = loss(self.config['alpha'], a, b, c, h, f, g, self.setup, domain, self.device)
                    loss_u = Loss.u(prediction_u, prediction_v, self.u_net, datau, datav, bdata)
                    self.av_l += loss_u
                    loss_u.backward(retain_graph=True)
                self.optimizer_u.step()
                if self.stop is not None and self.stop(self, points.interioru, domain):
                    torch.save(self.u_net.state_dict(), 'best_model_weights.pth')
                    print('Stopping Criterion Reached')
                    exit()

                if self.av_l.item() < self.best_l:
                    torch.save(self.u_net.state_dict(), 'best_model_weights.pth')

            for j in range(self.n2):
                self.optimizer_v.zero_grad()
                for (datau, datav, bdata) in points:
                    prediction_v = self.v_net(datav)
                    prediction_u = self.u_net(datau)
                    h, f, g, a, b, c = func_eval(datau.clone().detach(), bdata.clone().detach(), self.setup,
                                                 prediction_u, self.func_a, self.func_b, self.func_c, self.func_h,
                                                 self.func_f, self.func_g)
                    Loss = loss(self.config['alpha'], a, b, c, h, f, g, self.setup, domain, self.device)
                    loss_v = Loss.v(prediction_u, prediction_v, datau, datav)
                    loss_v.backward(retain_graph=True)
                self.optimizer_v.step()

            if report and k % report_it == 0:
                lu, lv = loss_u.item(), loss_v.item()
                print('iteration: ' + str(k), 'Loss u: ' + str(lu), 'Loss v: ' + str(lv))
                if self.func_u_sol is not None:
                    points = Comb_loader(self.setup['N_r'], self.setup['N_b'], domain, self.device)
                    L1 = L_norm(points.interioru, self.u_net, self.p, self.func_u_sol, domain.V(), self.setup['N_r']).item()
                    print('L^1 norm error: ' + str(L1))
                    # TODO: modify proj to support different number of plots
                    #proj(self.u_net, self.setup, k, self.device, axes=[0, 1], resolution=200, colours=20, save=False,
                         #show=show_plt, func_u_sol=self.func_u_sol)

            if k == self.iterations:
                print('Max Iterations Reached')

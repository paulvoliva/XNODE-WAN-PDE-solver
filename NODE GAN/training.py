from loss import loss
from model import NeuralODE, discriminator
import itertools
import torch
from itertools import product
from auxillary_funcs import proj, L_norm
from dataset import Comb_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def func_eval(X: torch.Tensor, BX: torch.Tensor, setup: dict, y_output_u: torch.Tensor, func_a, func_b, func_c, func_h, func_f, func_g):
    '''
    for computational efficiency the functions will be evaluated here

    Args:
        X: interior data points
        BX: boundary data points
        y_output_u: our guess for u  #He. I recomend to change y_output_u -> pred_u.
        func_: all the given functions to the PDE
    '''

    h = func_h(X[:, 0, :])
    f = func_f(X)
    g = func_g(BX)

    c = func_c(X, y_output_u)

    # a is meant to be a d by d-dimensional array with each element containing a tensor of size = [X.shape[0], X.shape[1]].
    a = torch.Tensor(setup['dim'], setup['dim'], X.shape[0], X.shape[1])

    for i, j in product(range(setup['dim']), repeat=2):
        a[i, j] = func_a(X, i, j)

    # b is meant to be a d-dimensional array with each element containing a tensor of size = [X.shape[0], X.shape[1]].
    b = torch.Tensor(setup['dim'], X.shape[0], X.shape[1])

    for i in range(setup['dim']):
        b[i] = func_b(X, i)

    return h.to(device), f.to(device), g.to(device), a.to(device), b.to(device), c.to(device)


def init_weights(layer):
    #He. this function ensures the correct initialization
    if type(layer) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)

 
#He. I suggest change NODE_solver -> XNODE_WAN_solvre
class NODE_solver:
    '''
    this class runs the XNODE_WAN algorithm

    Args:
        params: dict: a dictionary with our setup and config dicts as well as the number of iterations
        func_: all the funcs that are the coefficients and boundary values from the PDE
        domain: this is the initialised class object that will generate the points we need (more on this in dataset)
        func_u_sol: the solution to our PDE if known
        p: if fun_u_sol is given the norm difference of our guess to the true solution in L^p is computed
    '''
    def __init__(self, params: dict, func_a, func_b, func_c, func_h, func_f, func_g, domain, func_u_sol = None, p: float = 1):
        self.params = params
        self.func_a = func_a
        self.func_b = func_b
        self.func_c = func_c
        self.func_h = func_h
        self.func_f = func_f
        self.func_g = func_g
        self.domain = domain
        self.func_u_sol = func_u_sol
        self.p = p

        i = iter(params.items())
        self.config = dict(itertools.islice(i, 13))
        self.setup = dict(itertools.islice(i, 6))
        self.iterations = dict(i)['iterations']
        self.n1 = self.config['n1']
        self.n2 = self.config['n2']

        self.points = Comb_loader(self.setup['N_r'], self.setup['N_b'], self.domain) #He. I don't think this line is needed here.

        # seu up neural network models
        #He. I suugest change NeuralODE->XNODE and discriminator -> DNN_v (we did not mention discriminator in our paper).
        self.u_net = NeuralODE(self.config['u_hidden_dim'], 1, self.func_h, self.setup,
                               self.config['u_hidden_hidden_dim'], self.config['u_layers'], self.config['solver'],
                               self.config['min_steps'], self.config['adjoint']).to(device) #torch.nn.DataParallel().to(device)
        
        self.v_net = discriminator(self.config, self.setup).to(device) #torch.nn.DataParallel().to(device)
        
        #initialization
        self.u_net.apply(init_weights)
        self.v_net.apply(init_weights)

        # set up optimizers for v_net and u_net
        self.optimizer_u = torch.optim.Adam(self.u_net.parameters(), lr=self.config['u_rate'])
        self.optimizer_v = torch.optim.Adam(self.v_net.parameters(), lr=self.config['v_rate'])

    def train(self, report: bool = False, report_it: int = 10, show_plt: bool = False):
        '''
        Args:
            report: whether or not to display information on the progress of the algorithm
            report_it: after how many iterations to report
            show_plt: whether or not to show a plot
        '''
        for k in range(self.iterations):
            points = Comb_loader(self.setup['N_r'], self.setup['N_b'], self.domain) #He. what does Comb mean here?

            for i in range(self.n1):
                 for (datau, datav, bdata) in points:
                    self.optimizer_u.zero_grad()
                    datau, datav, bdata = datau.requires_grad_(True), datav.requires_grad_(True), bdata.requires_grad_(True)
                    prediction_v = self.v_net(datav)
                    prediction_u = self.u_net(datau)
                    h, f, g, a, b, c = func_eval(datau.clone().detach(), bdata.clone().detach(), self.setup, prediction_u, self.func_a, self.func_b,
                                                 self.func_c, self.func_h, self.func_f, self.func_g)
                    Loss = loss(self.config['alpha'], a, b, c, h, f, g, self.setup, self.domain)
                    loss_u = Loss.u(prediction_u, prediction_v, self.u_net, datau, datav, bdata)
                    loss_u.backward(retain_graph = True)
                    self.optimizer_u.step()

            for j in range(self.n2):
                for (datau, datav, bdata) in points:
                    self.optimizer_v.zero_grad()
                    prediction_v = self.v_net(datav)
                    prediction_u = self.u_net(datau)
                    h, f, g, a, b, c = func_eval(datau.clone().detach(), bdata.clone().detach(), self.setup, prediction_u, self.func_a, self.func_b, self.func_c, self.func_h, self.func_f,
                                                 self.func_g)
                    Loss = loss(self.config['alpha'], a, b, c, h, f, g, self.setup, self.domain)
                    loss_v = Loss.v(prediction_u, prediction_v, datau, datav)
                    loss_v.backward(retain_graph=True)
                    self.optimizer_v.step()

            if report and k % report_it == 0:
                lu, lv = loss_u.item(), loss_v.item()
                print('iteration: ' + str(k), 'Loss u: ' + str(lu), 'Loss v: ' + str(lv))
                if self.func_u_sol != None:
                    L1 = L_norm(points.interioru, prediction_u, self.p, self.func_u_sol, self.domain.V()).item()
                    print('L^1 norm error: ' + str(L1))
                    # TODO: modify proj to support different number of plots
                    proj(self.u_net, axes=[0, 1], resolution=200, colours=20, iteration=k, save=False, show=show_plt, func_u_sol=self.func_u_sol)

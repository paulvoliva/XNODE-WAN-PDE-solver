from src.training import NODE_WAN_solver
import torch
import argparse
import yaml
from src.dataset import *
import importlib

'''
This is the document that allows to solve the PDEs and to interact with the algorithm.
'''

'''
# General Form of our problem:
\begin{equation}
\left\{\begin{array}{ll}
u_{t}-\sum_{i=1}^d \partial_i (\sum_{j=1}^d a_{ij} \partial_j u) + \sum_{i=1}^d b_i \partial_i u + cu = f, & \text { in } \Omega \times[T0, T] \\
u(x, t)=g(x, t), & \text { on } \partial \Omega \times[T0, T] \\
u(x, 0)=h(x), & \text { in } \Omega
\end{array}\right.
\end{equation}
where $x$ is a d-dimensional vector. You specify these functions just below
'''

parser = argparse.ArgumentParser(prog='XNODE-WAN PDE solver',
                                 description='a general purpose parabolic PDE solver using the XNODE-WAN architecture')

parser.add_argument('-w', '--work_dir', type=str, default='./', help='directory for the best model parameters')

parser.add_argument('--params', help='an experiment setup to load', required=True)
parser.add_argument('--funcs', help='location of the functions for the PDE (omit .py)', required=True)
parser.add_argument('--device', default=None, help='the device to run the experiment, default is cuda if available')
parser.add_argument('--report', type=bool, default=True, help='boolean for reporting the progress')
parser.add_argument('--report_it', type=int, default=10, help='number of iterations between reporting progress')
parser.add_argument('--show_plt', type=bool, default=False, help='whether to show plots at report_it intervals')

args = parser.parse_args(['--params', 'cube_pde.yaml', '--funcs', 'Ex4_1_funcs'])

funcs = importlib.import_module('configs.'+args.funcs)
names = [x for x in funcs.__dict__ if not x.startswith("_")]
globals().update({k: getattr(funcs, k) for k in names[2:]})

with open('NODE_GAN working/configs/'+args.params, 'r') as parameters:
    params = yaml.safe_load(parameters)

# setting to cuda

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device is None else args.device


if __name__ == '__main__':
    solver = NODE_WAN_solver(params, func_a, func_b, func_c, func_h, func_f, func_g, device, args.work_dir, func_u_sol=func_u_sol, p=2, stop=stop)
    solver.train(report=args.report, report_it=args.report_it, show_plt=args.show_plt)

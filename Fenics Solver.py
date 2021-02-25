from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import torch
from itertools import product

class Solver():
    '''
    This class is used to solve the PDE numerically. It can be initialised and then solved and after this we can
    call the different time points it has computed.
    '''
    def __init__(self,
                 timesteps,
                 mesh_size,
                 f,
                 g,
                 h,
                 a,
                 b,
                 c,
                 domain_vertices,
                 boundary_condition = 'Dirichlet',
                 rectangular_mesh = True,
                 domain_mesh = None,
                 T = 1.0):
        '''
        timesteps (int): The number of time intervals
        mesh_size (int): The mesh size (make None if the domain is not a rectangle)
        f (str): f in terms of C/C++ code
        g (str): g in terms of C/C++ code
        h (str): h in terms of C/C++ code
        a (tensor): The a matrix coefficient (for this, b and c write the function in terms of Expression objects)
        b (tensor): The b vector coefficient
        c : The function c
        domain_vertices (tuple): The vertices of the rectangle for the mesh in the form of tuples themselves (input none
            if the domain is not a rectangle)
        boundary_condition (str): Either 'Dirichlet' or 'Neumann'
        rectangular_mesh (boolean): Wether the domain is rectangular
        domain_mesh (__dolfin_cpp_mesh.Mesh): A dolfin mesh that is our domain
        T (int): The final time point
        '''
        super().__init__()
        self.timesteps = timesteps
        self.T = T
        self.dt = T/timesteps
        self.mesh_size = mesh_size
        self.f = Expression(f, t=0, degree=2)
        self.g = Expression(g, t=0, degree=2)
        self.h = Expression(h, degree=2)
        self.dim = a.shape[0]
        self.a = a.numpy()
        self.b = b.numpy()
        self.c = c
        self.domain_vertices = domain_vertices
        self.boundary_condition = boundary_condition
        self.rectangular_mesh = rectangular_mesh
        self.domain_mesh = domain_mesh
        self.T = T

        if self.rectangular_mesh:
            bottom, top = self.domain_vertices
            self.mesh = RectangleMesh(Point(bottom), Point(top), self.mesh_size, self.mesh_size)
        else:
            self.mesh = self.domain_mesh

        self.V = FunctionSpace(self.mesh, 'P', 2)

        def boundary(x, on_boundary):
            return on_boundary

        if self.boundary_condition == 'Dirichlet':
            self.bc = DirichletBC(self.V, self.g, boundary)

        self.u_n = interpolate(self.h, self.V)
        self.u = Function(self.V)
        self.v = TestFunction(self.V)

        self.F = self.u*self.v*dx - self.u_n*self.v*dx - self.dt*self.f*self.v*dx + self.dt*eval(self.c)*self.u*self.v*dx

        for i,j in product(range(self.dim), repeat=2):
            self.F += self.dt*self.a[i,j]*self.u.dx(j)*self.v.dx(i)*dx(domain=self.mesh)

        for i in range(self.dim):
            self.F += self.dt*self.b[i,0]*self.u.dx(i)*dx(domain=self.mesh)

        self.solutions = {'u_0': self.u_n}

    def compute(self):
        '''
        This section just solves the PDE
        '''

        t = 0
        for n in range(self.timesteps):
            # Update current time
            t += self.dt
            self.g.t = t
            self.f.t = t

            # Compute solution
            solve(self.F == 0, self.u, self.bc)

            self.solutions['u_'+str(n+1)] = self.u

    def call(self, points):
        '''
        points (tensor): tensor containing the point values and time values in form [N, L, dim+1]
        '''

        N = points.shape[0]
        L = points.shape[1]

        time_ids = self.timesteps*points[:, :, 0]/self.T

        sol = torch.zeros(N, L)

        for i in range(N):
            for j in range(L):
                x = points[i, j, 1].numpy()
                y = points[i, j, 2].numpy()
                t = int(time_ids[i, j].numpy())
                sol[i, j] = self.solutions['u_'+str(t)](x, y)

        return sol


'''
a = torch.Tensor([[1, 0], [0, 1]])
b = torch.zeros(2,1)
f = '(pow(pi,2)-2)*sin(pi/2*x[0])*cos(pi/2*x[1])*exp(-t)-4*pow(sin(pi/2*x[0])*cos(pi/2*x[1])*exp(-t),2)'
g = '2*sin(pi/2*x[0])*cos(pi/2*x[1])*exp(-t)'
h = '2*sin(pi/2*x[0])*cos(pi/2*x[1])'

num_sol = Solver(5, 50, f, g, h, a, b, '-self.u', ((-1, -1),(1, 1)))
num_sol.compute()
pts = torch.zeros(2,2,3)
pts[1,1,0] = 0.33331
print(num_sol.call(pts))
'''
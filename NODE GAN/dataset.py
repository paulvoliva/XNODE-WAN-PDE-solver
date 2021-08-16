"""
dataset.py
=============================
Dataset classes and domain shapes
"""
import torch
from torch.utils.data import Dataset
import numpy as np


def fillt(inputs: torch.Tensor, setup: dict, min_steps: int = 5):
    '''
    this function extends the timeseries and extends it to ensure there are at least min_steps in the time interval
    '''
    out = torch.linspace(setup['T0'], setup['T'], min_steps).to(inputs.device)

    if len(inputs.shape) == 2:
        ind = int((inputs[0, 0] * min_steps).item())
        out[ind] = inputs[0, 0]
        return out

    for i in range(inputs.shape[1]):
        ind = int((inputs[0, i, 0] * min_steps).item())
        out[ind] = inputs[0, i, 0]
        return out

'''
For shapes we need to create a class with the functions:
interior(N_r):  takes in an int and returns a list of different groups of paths in the domain (each group needs to have 
                equally long paths)
boundary(N_b):  takes in an int and returns a list of different groups of paths on the boundary (each group needs to 
                have equally long paths)
func_w(x):      takes in a point and measures how far it is from the boundary
V():            returns a float that is the volume of the domain
'''

class NSphere_Tcone:
    '''
    this class samples points in a hypersphere of radius r*(1-t)
    Note this is a time-dependent domain.
    Args:
        r: radius at time T0 of hypersphere
    '''
    def __init__(self, r: float, dim: int, T0: float, T: float, N_t: int):
        self.r = r
        self.dim = dim
        self.T0 = T0
        self.T = T
        self.N_t = N_t
        self.times, i = torch.sort(torch.Tensor(self.N_t).uniform_(T0, T), 0)
        self.times[0], self.times[-1] = T0, T
    
    #He. Some explaination here is needed. I am lost.
    def surf(self, N: int):
        normal_deviates = np.random.normal(size=(self.dim, N))

        radius = np.sqrt((normal_deviates ** 2).sum(axis=0))
        return normal_deviates / radius

    def interior(self, N_r: int):
        points = self.surf(N_r)
        points *= np.random.rand(N_r) ** (1 / self.dim)
        time_data = self.times.repeat(N_r, 1).unsqueeze(2)

        datapoints = []
        k = self.N_t

        for t in self.times.numpy()[::-1]:
            idx = np.sqrt(np.sum(points ** 2, 0)) < (1 - t)
            npoints = torch.transpose(torch.from_numpy(points[:, idx]), 0, 1).unsqueeze(1).repeat(1, k, 1)
            points = np.delete(points, idx, 1)
            if npoints.shape[0] != 0:
                datapoints.append(torch.cat((time_data[:npoints.shape[0], :k], npoints), 2).requires_grad_(True))
            k -= 1

        return datapoints[::-1]

    def boundary(self, N_b: int):
        datapoints = []

        for t in self.times.numpy():
            n = int(N_b * (1 - t) ** self.dim)
            points = self.surf(n) * (1 - t)
            points = torch.transpose(torch.from_numpy(points), 0, 1).unsqueeze(1)
            ones = torch.ones(n, 1, 1)
            if n != 0:
                datapoints.append(torch.cat((t * ones, points), 2).requires_grad_(True))

        return datapoints

    def func_w(self, x: torch.Tensor):
        dist = torch.sqrt(torch.sum(x[:, :, 1:] ** 2, 2))
        return self.r * (1 - x[:, :, 0]) - dist

    def V(self):
        from scipy.special import gamma
        import math
        timecomp = ((1 - self.T0) ** (self.dim + 1) / (self.dim + 1) - (1 - self.T) ** (self.dim + 1) / (self.dim + 1))
        return math.pi ** (self.dim / 2) / gamma(self.dim / 2 + 1) * self.r ** self.dim * timecomp


class Hypercube:
    '''
    this class samples points in a hypercube from (bot,..., bot) to (top,...,top)
    Note that the domain is time-independent.

    Args:
        top: float, the value of the top coordinate of the cube
        bot: float, the value of the bottom coordinate of the cube
    '''
    def __init__(self, top: float, bot: float, dim: int, T0: float, T: float, N_t: int):
        assert top > bot, "The hypercube needs to have volume"
        self.top = top
        self.bot = bot
        self.dim = dim
        self.T0 = T0
        self.T = T
        self.N_t = N_t
        self.times, i = torch.sort(torch.Tensor(self.N_t).uniform_(T0, T), 0)
        self.times[0], self.times[-1] = T0, T

    def interior(self, N_r: int):
        #init = torch.Tensor(N_r, 1, self.dim).uniform_(self.bot, self.top)
        #x = init.repeat(1, self.N_t, 1)
        #xt = torch.cat((self.times.unsqueeze(1).repeat(N_r, 1, 1), x), dim=2)
        
        #He. I suggest change the above three lines to the bottom for better clarity and consistency.
        x = torch.Tensor(N_r, 1, self.dim).uniform_(self.bot, self.top).repeat(1, self.N_t, 1) #He. x.size = [N_r, N_t, self.dim]
        t = self.times.unsqueeze(1).repeat(N_r, 1, 1) #He. t.size = [N_r, N_t, 1]
        xt = torch.cat((t, x), dim=2)
        return xt #He. xt.size = [N_r, N_t, self.dim + 1]

    def boundary(self, N_b: int):
        x = torch.Tensor(N_b, 1, self.dim).uniform_(self.bot, self.top).repeat(1, self.N_t, 1)  #He. x.size = [N_b, N_t, self.dim]
        t = self.times.unsqueeze(1).repeat(N_b, 1, 1) #He. t.size = [N_b, N_t, 1]
        xt = torch.cat((t, x), dim=2) #He. xt.size = [N_b, N_t, self.dim + 1]
        
        tops = self.top * torch.ones(1, self.N_t, 1)
        bots = self.bot * torch.ones(1, self.N_t, 1)
        rand = torch.Tensor(N_b, 1, self.dim).uniform_(self.bot, self.top)

        n = int(N_b / self.dim / 2) #He. do you need to check at the beginning if N_b is a multiplier of self.dim*2?
        num = [n * i for i in range(2 * self.dim)]
        num[0] = 0
        num.append(N_b)
       
        for i in range(self.dim):
            xt[num[2 * i]:num[2 * i + 1], :, i + 1] = tops.repeat(num[2 * i + 1] - num[2 * i], 1, 1).squeeze()
            xt[num[2 * i + 1]:num[2 * i + 2], :, i + 1] = bots.repeat(num[2 * i + 2] - num[2 * i + 1], 1, 1).squeeze()
        
        idx = torch.randperm(N_b) #He. permute the boundary points.
        
        return xt[idx]

    def func_w(self, x: torch.Tensor):
        #He. what is the purpose of this function?
        disttop = torch.min(torch.abs(self.top - x), dim=2).values
        distbot = torch.min(torch.abs(self.bot - x), dim=2).values
        dist = torch.minimum(disttop, distbot)
        return dist

    def V(self):
        #He. what is the purpose of this function?
        return (self.top - self.bot) ** self.dim * (self.T - self.T0)


class Comb_loader(Dataset):
    '''
    a wrapper for our shapes so that they return equally long paths in batches

    Args:
        shape: the shape that we will use as our domain as specified above
    '''
    def __init__(self, N_r: int, N_b: int, shape):
        self.N_r = N_r
        self.N_b = N_b
        self.shape = shape
        self.interioru = self.shape.interior(self.N_r)
        self.interiorv = self.shape.interior(self.N_r) #He.if interioru = interiorv, why do we need to differentiate them? 
        #He. For the v_net, I think it is better not to use the strucured sampling and instead we should use the random sampling.
        #TODO: add a seperate dataloader class for v_net may be necessary.
        self.boundary = self.shape.boundary(self.N_b)  #He. Here we don't differentiate boundary points for u and v.
       
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        m = min(len(self.interioru), len(self.interiorv)) 
        
        #He. what are we doing below? change the tensor into a list? why take the minimal?
        self.interior_u = [self.interioru[i] for i in range(m) if
                           self.interiorv[i].shape[1] == self.interioru[i].shape[1]] if isinstance(self.interioru, list) else self.interioru
        #He. isinstance(self.interioru, list) should be false since self.interioru is a tensor? I am totally lost here.
        self.interior_v = [self.interiorv[i] for i in range(m) if
                           self.interiorv[i].shape[1] == self.interioru[i].shape[1]] if isinstance(self.interiorv, list) else self.interiorv

    def __len__(self):
        return len(self.interior_u) if isinstance(self.interior_u, list) else 1

    def __getitem__(self, idx):
        if not isinstance(self.interioru, list) and idx != 0:
            raise IndexError
        data_u = self.interior_u[idx] if isinstance(self.interioru, list) else self.interior_u
        data_v = self.interior_v[idx] if isinstance(self.interioru, list) else self.interior_v
        boundary_ = self.boundary[idx] if isinstance(self.interioru, list) else self.boundary
        m = min(len(data_v), len(data_u))  #He. Why take the minimal again??
        r = (data_u[:m].to(self.device).requires_grad_(True), data_v[:m].to(self.device).requires_grad_(True), boundary_.to(self.device).requires_grad_(True))
        return r

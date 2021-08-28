"""
dataset.py
=============================
Dataset classes and domain shapes
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from itertools import groupby
import math


def fillt(inputs: torch.Tensor, T: float, T0: float, min_steps: int = 5):
    '''
    this function extends the timeseries and extends it to ensure there are at least min_steps in the time interval
    '''
    times = inputs
    min_step_ = (T - T0) / min_steps
    diffs = torch.cat((torch.tensor(min_step_/2).view(1,).to(inputs.device), torch.abs(times[:-1]-times[1:])), 0)

    idx = torch.nonzero(diffs > min_step_).squeeze(1)
    idx = torch.cat((idx, torch.tensor([times.shape[0]]).to(inputs.device)), dim=0).int()
    i = torch.tensor(range(times.shape[0])).to(inputs.device)
    out = times[0].view(1,)

    for k in range(idx.shape[0] - 1):
        n = round((times[idx[k]].item()-2*min_step_-out[-1].item())/min_step_) + 1
        fill = torch.linspace(out[-1].item()+min_step_, times[idx[k]].item()-min_step_, n).to(inputs.device)
        i[idx[k]:] += fill.shape[0]
        out = torch.cat((out, fill, (times[idx[k]:idx[k+1]]).to(inputs.device)), dim=0)

    return i, out

'''
For shapes we need to create a class with the functions:
interior(N_r):  takes in an int and returns a list of different groups of paths in the domain (each group needs to have 
                equally long paths)
boundary(N_b):  takes in an int and returns a list of different groups of paths on the boundary (each group needs to 
                have equally long paths)
func_w(x):      takes in a point and measures how far it is from the boundary
bound_pad(x):   takes a point in the domain and replaces the time coordinate with the one which places x on the nearest 
                boundary point (it needs to return an index for the paths (None if there is only one length of path), 
                one for the times that we input in the timesteps and the filled timesteps)
V():            returns a float that is the volume of the domain
'''


class NSphere_THourglass:
    '''
    this class samples points in a hypersphere of radius r*((T-T0)-t) for t<(T-T0)/2 and r*t for t>(T-T0)/2

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

    def surf(self, N: int):
        normal_deviates = np.random.normal(size=(self.dim, N))

        radius = np.sqrt((normal_deviates ** 2).sum(axis=0))
        return self.r * normal_deviates / radius

    def split_l(self, idx: torch.Tensor, i: int):
        F_pos = torch.nonzero(idx[i] == False)
        if F_pos.shape[0] != 0:
            return [F_pos[0], idx.shape[1]-F_pos[-1]-1]
        else:
            return idx.shape[1]

    def pad(self, x: torch.Tensor):
        t = (torch.sqrt(torch.sum(x[0, 1:]**2, 0))/self.r).view(1, 1)
        return torch.cat((torch.cat((t, x[0, 1:].unsqueeze(0)), dim=1), x), dim=0)

    def interior(self, N_r: int):
        points = self.surf(N_r)
        points *= np.random.rand(N_r) ** (1 / self.dim)
        time_data = self.times.repeat(N_r, 1).unsqueeze(2)

        points = torch.transpose(torch.from_numpy(points), 0, 1).unsqueeze(1).repeat(1, self.N_t, 1)
        bound = torch.zeros_like(points[:, :, 0]).unsqueeze(2)
        i = torch.le(time_data, (self.T-self.T0)/2)
        bound[i] = self.r*((self.T-self.T0) - time_data)[i].double()
        bound[~i] = self.r*time_data[~i].double()
        idx = (torch.sqrt(torch.sum(points ** 2, 2)).unsqueeze(2) < bound).squeeze()
        points = torch.cat((time_data, points), dim=2)
        data_ = [points[k, idx[k]].split(self.split_l(idx, k), dim=0) for k in range(points.shape[0])]
        l1, l2 = zip(*[(k+(None,))[:2] for k in data_])
        l2 = list(filter(None.__ne__, l2))
        l2 = [self.pad(k) for k in l2]
        l_1, l_2 = [i.unsqueeze(0) for i in l1], [i.unsqueeze(0) for i in l2]
        data_1, data_2 = sorted(l_1, key=lambda x: x.shape[1]), sorted(l_2, key=lambda x: x.shape[1])
        data1, data2 = list(tuple(r) for k, r in groupby(data_1, key=lambda x: x.shape[1])), list(tuple(r) for k, r in
                                                                                                  groupby(data_2, key=
                                                                                                  lambda x: x.shape[1]))
        data1, data2 = [torch.cat(k, dim=0) for k in data1], [torch.cat(k, dim=0) for k in data2]
        datapoints = sorted([*data1, *data2], key=lambda x: x.shape[1])
        return datapoints

    def boundary(self, N_b: int):
        datapoints = []

        for t in self.times.numpy():
            n = int(N_b * ((self.T-self.T0) - t) ** self.dim) if t < (self.T-self.T0)/2 else int(N_b * t ** self.dim)
            points = self.surf(n) * ((self.T-self.T0) - t) if t < (self.T-self.T0)/2 else self.surf(n) * t
            points = torch.transpose(torch.from_numpy(points), 0, 1).unsqueeze(1)
            ones = torch.ones(n, 1, 1)
            if n != 0:
                datapoints.append(torch.cat((t * ones, points), 2).requires_grad_(True))

        return datapoints

    def func_w(self, x: torch.Tensor):
        res = torch.ones_like(x[:, :, 0])
        dist = torch.sqrt(torch.sum(x[:, :, 1:] ** 2, 2))
        idx = torch.le(x[:, :, 0], (self.T-self.T0)/2)
        res[idx] = self.r * ((self.T-self.T0) - x[:, :, 0][idx]) - dist[idx]
        res[~idx] = self.r * x[:, :, 0][~idx] - dist[~idx]
        return res

    def bound_pad(self, x):
        t = x[0, 0, 0]
        t_ = self.T0*torch.ones_like(x[:, 0, 0]).to(x.device) if t<(self.T-self.T0)/2 else None
        i = None
        if t_ is None:
            r = torch.sqrt(torch.sum(x[:, 0, 1:]**2, dim=-1))
            idx = torch.le(r, self.r*(self.T-self.T0)/2)
            t_ = torch.zeros_like(x[:, 0, 0]).to(x.device)
            t_[idx] = self.T0
            t_[~idx] = r[~idx]/self.r
            i = torch.le(self.func_w(x[:, 0].unsqueeze(1)), 1e-5).squeeze().int()
        t_ = torch.cat((t_.unsqueeze(1), x[:, :, 0]), dim=1)
        data = [fillt(t_[k], self.T, self.T0, self.N_t)[1] for k in range(t_.shape[0])]
        idx = [fillt(t_[k], self.T, self.T0, self.N_t)[0] for k in range(t_.shape[0])] if i is None else [fillt(t_[k][i[k]:], self.T, self.T0, self.N_t)[0][~i[k]+2:] for k in range(t_.shape[0])]
        i = sorted(range(len(data)), key=lambda k: data[k].shape[0])
        data = [data[k] for k in i]
        idx = [idx[k] for k in i]
        idx_ = [(k, l) for k, l in zip(idx, data)]
        data = [(next(r), len(tuple(r))+1) for k, r in groupby(data, key=lambda k: k.shape[0])]
        idx = [next(r)[0] for k, r in groupby(idx_, key=lambda k: k[1].shape[0])]
        lst = [k[1] for k in data]
        data = [k[0] for k in data]
        lst = list(np.cumsum(lst))
        lst.insert(0, 0)
        path_i = [torch.tensor(i[lst[k]:lst[k+1]]) for k in range(len(lst)-1)]
        return path_i, idx, data

    def V(self):
        from scipy.special import gamma
        import math
        timecomp = 2*((1 - self.T0) ** (self.dim + 1) / (self.dim + 1) - (1 - (self.T-self.T0)/2) ** (self.dim + 1) /
                    (self.dim + 1))
        return math.pi ** (self.dim / 2) / gamma(self.dim / 2 + 1) * self.r ** self.dim * timecomp


class NSphere_TCone:
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

    def surf(self, N: int):
        normal_deviates = np.random.normal(size=(self.dim, N))

        radius = np.sqrt((normal_deviates ** 2).sum(axis=0))
        return self.r * normal_deviates / radius

    def interior(self, N_r: int):
        points = self.surf(N_r)
        points *= np.random.rand(N_r) ** (1 / self.dim)
        time_data = self.times.repeat(N_r, 1).unsqueeze(2)

        datapoints = []
        k = self.N_t

        for t in self.times.numpy()[::-1]:
            idx = np.sqrt(np.sum(points ** 2, 0)) < self.r*(1 - t)
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

    def bound_pad(self, x):
        t = torch.cat((torch.tensor(self.T0).view(1), x[0, :, 0]), dim=0)
        idx, data = fillt(t, self.T, self.T0, self.N_t)
        return None, idx[1:], data

    def V(self):
        from scipy.special import gamma
        import math
        timecomp = ((1 - self.T0) ** (self.dim + 1) / (self.dim + 1) - (1 - self.T) ** (self.dim + 1) / (self.dim + 1))
        return math.pi ** (self.dim / 2) / gamma(self.dim / 2 + 1) * self.r ** self.dim * timecomp


class Hypercube:
    '''
    this class samples points in a hypercube from (bot,..., bot) to (top,...,top)

    Note this is a time-independent domain.
    Args:
        top_bot: a tuple of the form (bot, top)
    '''
    def __init__(self, top_bot: tuple, dim: int, T0: float, T: float, N_t: int):
        assert top_bot[1] > top_bot[0], "The hypercube needs to have volume"
        self.top = top_bot[1]
        self.bot = top_bot[0]
        self.dim = dim
        self.T0 = T0
        self.T = T
        self.N_t = N_t
        self.times, i = torch.sort(torch.Tensor(self.N_t).uniform_(T0, T), 0)
        self.times[0], self.times[-1] = T0, T

    def interior(self, N_r: int):
        x = torch.Tensor(N_r, 1, self.dim).uniform_(self.bot, self.top).repeat(1, self.N_t, 1)
        t = self.times.unsqueeze(1).repeat(N_r, 1, 1)
        xt = torch.cat((t, x), dim=2)
        return xt

    def boundary(self, N_b: int):
        t = self.times.unsqueeze(1).repeat(N_b, 1, 1)
        x = torch.Tensor(N_b, 1, self.dim).uniform_(self.bot, self.top).repeat(1, self.N_t, 1)
        xt = torch.cat((t, x), dim=2)
        tops = self.top * torch.ones(1, self.N_t, 1)
        bots = self.bot * torch.ones(1, self.N_t, 1)
        rand = torch.Tensor(N_b, 1, self.dim).uniform_(self.bot, self.top)

        n = int(N_b / self.dim / 2)
        num = [n * i for i in range(2 * self.dim)]
        num[0] = 0
        num.append(N_b)

        for i in range(self.dim):
            xt[num[2 * i]:num[2 * i + 1], :, i + 1] = tops.repeat(num[2 * i + 1] - num[2 * i], 1, 1).squeeze()
            xt[num[2 * i + 1]:num[2 * i + 2], :, i + 1] = bots.repeat(num[2 * i + 2] - num[2 * i + 1], 1, 1).squeeze()

        idx = torch.randperm(N_b)

        return xt[idx]

    def func_w(self, x: torch.Tensor):
        disttop = torch.min(torch.abs(self.top - x[:, :, 1:]), dim=2).values
        distbot = torch.min(torch.abs(self.bot - x[:, :, 1:]), dim=2).values
        dist = torch.minimum(disttop, distbot)
        return dist

    def bound_pad(self, x):
        t = torch.cat((torch.tensor(self.T0).view(1), x[0, :, 0]), dim=0)
        idx, data = fillt(t, self.T, self.T0, self.N_t)
        return None, idx[1:], data

    def V(self):
        return (self.top - self.bot) ** self.dim * (self.T - self.T0)


class Comb_loader(Dataset):
    '''
    a wrapper for our shapes so that they return equally long paths in batches

    Args:
        shape: the shape that we will use as our domain as specified above
    '''
    def __init__(self, N_r: int, N_b: int, shape, device):
        self.N_r = N_r
        self.N_b = N_b
        self.shape = shape
        self.device = device

        interior = self.shape.interior(self.N_r)
        self.interioru = [i.requires_grad_(True) for i in interior] if isinstance(interior, list) else interior.requires_grad_(True)
        self.interiorv = [i.clone().detach().requires_grad_(True) for i in self.interioru] if isinstance(self.interioru, list) else self.shape.interior(self.N_r).clone().detach().requires_grad_(True)
        boundary = self.shape.boundary(self.N_b)
        self.boundary = [i.requires_grad_(True) for i in boundary] if isinstance(boundary, list) else boundary.requires_grad_(True)

    def __len__(self):
        return len(self.interioru) if isinstance(self.interioru, list) else 1

    def __getitem__(self, idx):
        if not isinstance(self.interioru, list) and idx != 0:
            raise IndexError
        data_u = self.interioru[idx] if isinstance(self.interioru, list) else self.interioru
        data_v = self.interiorv[idx] if isinstance(self.interioru, list) else self.interiorv
        boundary_ = self.boundary[idx] if isinstance(self.interioru, list) else self.boundary
        r = (data_u.to(self.device), data_v.to(self.device), boundary_.to(self.device))
        return r

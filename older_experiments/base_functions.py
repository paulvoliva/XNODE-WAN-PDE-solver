import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


## time step
def time_grids(t0,T,N):
  """
        outputing equal time grids between t0 and T with stepsize (T-t0)/N
  """
  return t0+torch.arange(N+1)*(T-t0)/N

def func_f(x, t):
    """
        the value of force term f
        
        (r.f. Eqn.(20) )
    """
    lens=x.shape[0]
    f=torch.Tensor(lens,1)
    
    a1=torch.sin(x[:,0]*math.pi/2)
    a2=torch.cos(x[:,1]*math.pi/2)
    a3=math.exp(-t)
    
    f[:,0]=((math.pi**2)-2)*a1*a2*a3 - 4*(a1**2)*a2*(a3**2)
    
    return(f)

def func_u(x, t):
    """
        the true sol for pde, meanwhile, also 
        boundary condition for all t, eg, u(x,y,t) at domain boundary
    """
    lens = x.shape[0]
    u = torch.Tensor(lens,1)
    u[:,0] = 2 * torch.sin(x[:,0] * math.pi/2) * torch.cos(x[:,1]*math.pi/2) * math.exp(-t)
    return(u) 

def func_h(x):
    """
        initial condition for all x,y, eg, u(x,y,0) 
                (r.f. Eqn.(20) )
    """
    lens = x.shape[0]
    h = torch.Tensor(lens,1)
    h[:,0] = 2 * torch.sin(x[:,0] * math.pi/2) * torch.cos(x[:,1]*math.pi/2)
    return(h)

def func_d2u_0(x_test_):
    """
       computing laplacian of u_theta at initial step, i.e., t=h
        
    """
    lens=x_test_.shape[0]
    h = func_h(x_test_)
    h.backward(torch.ones([lens,1]),retain_graph=True)
    h.backward(torch.ones([lens,1]))
    d2u=x_test_.grad
    d2u_=torch.Tensor(lens,1)
    d2u_[:,0]=d2u[:,0]+d2u[:,1]
    
    return (d2u_)

def func_d2u_1(x_test,x_test_,net_u,optimizer_u):
    """
       computing laplacian of u_theta at all steps after the initial step, i.e., t=ih, i>1
        
    """
    lens = x_test.shape[0]
    output_1=net_u(x_test_)
    optimizer_u.zero_grad()
    output_1.backward(torch.ones([lens,1]), retain_graph=True)
    output_1.backward(torch.ones([lens,1]), retain_graph=True)
    d2u_2=x_test_.grad
    d2u_2_=d2u_2[:,0]+d2u_2[:,1]
    
    return (d2u_2_)


def func_new_f_0(x_test_, h,t):
    """
        new force term after applying Crank-Nicolson scheme at initial step, i.e., t=h
        (r.f. paragraph after Eqn. (11) or Eqn. (IV.16) in MSc student's disertation )
    """
    lens = x_test_.shape[0]
    new_f = torch.Tensor(lens,1)
    
    t_previous=t-h
    
    new_f = func_u(x_test_,t_previous) + (func_f(x_test_, t) + func_d2u_0(x_test_)\
            + func_u(x_test_,t_previous)**2 + func_f(x_test_,t_previous)) * h/2
    
    return(new_f)

def func_new_f_1(x_test,x_test_, h,t,y_output_u,net_u,optimizer_u):
    """
        new force term after applying Crank-Nicolson scheme after initial step,i.e, t=ih, i>1
        (r.f. paragraph after Eqn. (11) or Eqn. (IV.16) in MSc student's disertation )
    """
    lens = x_test.shape[0]
    new_f = torch.Tensor(lens,1)
    t_previous=t-h
    
    new_f = y_output_u + (func_f(x_test, t) + func_d2u_1(x_test,x_test_,net_u,optimizer_u) \
                            + 2*y_output_u**2 + func_f(x_test,t_previous)) * h/2
    
    return(new_f)


# w_val
def func_w(x):
    """
        w part of test function, with boundary being zero
    """
    lens = x.shape[0]
    w_bool = torch.gt(1 - torch.abs(x[:,0]), torch.zeros(lens)) & torch.gt(torch.abs(x[:,0]), torch.zeros(lens))
    w_val = torch.where(w_bool, 1 - torch.abs(x[:,0]) + torch.abs(x[:,0]), torch.zeros(lens))
    w_val_ = torch.reshape(w_val,(lens,1))
    return (w_val_)


# grad_u
def grad_u(optimizer_u,x_test,x_test_,y_output_u):
    """
        computing gradient of sol function u_theta
    """
    optimizer_u.zero_grad()
    lens = x_test.shape[0]
    y_output_u.backward(torch.ones([lens,1]), retain_graph=True)
    grad_u = x_test_.grad
    return(grad_u) 

# grad_phi
def grad_phi(optimizer_v,x_test, x_test_, y_output_v):
    """
        computing gradient of test function phi
    """    
    optimizer_v.zero_grad()
    lens = x_test.shape[0]
    w = torch.reshape(func_w(x_test_), (lens, 1))
    phi = y_output_v.mul(w) #this is test function as w*v
    phi.backward(torch.ones([lens, 1]), retain_graph=True)
    grad_phi = x_test_.grad
    return (grad_phi)

# loss function
def I_0(optimizer_u,x_test,x_test_,y_output_u, optimizer_v,y_output_v,h,t):  
    """
        main part of the loss function at interior points at initial step,i.e, t=h
         (indeed the numerator part of Eqn. (4) ) 
        
    """    
    temp_grad_u=grad_u(optimizer_u,x_test,x_test_,y_output_u)
    
    l1 = (-h/2)*torch.sum(torch.mm(torch.transpose(temp_grad_u,0,1),grad_phi(optimizer_v,x_test, x_test_,y_output_v)))\
         +torch.sum(torch.mul((1-y_output_u*h/2),torch.mul(y_output_u,torch.mul(y_output_v, func_w(x_test)))))
        
    l = torch.sum(func_new_f_0(x_test_,h,t)*y_output_v*func_w(x_test))
    I = l1-l
    
    return(I)

def I_1(net_u,optimizer_u,x_test,x_test_,y_output_u, optimizer_v,y_output_v,h,t):
    """
        main part of the loss function at interior points after the initial step,i.e, t=ih, i>1
                 (indeed the numerator part of Eqn. (4) ) 
    """
    temp_grad_u=grad_u(optimizer_u,x_test,x_test_,y_output_u)
    
    l1 = (-h/2)*torch.sum(torch.mm(torch.transpose(temp_grad_u,0,1),grad_phi(optimizer_v,x_test, x_test_,y_output_v)))\
    +torch.sum((1-y_output_u*h/2)*y_output_u*y_output_v*func_w(x_test))
    
    l = torch.sum(func_new_f_1(x_test,x_test_, h,t,y_output_u,net_u,optimizer_u)*y_output_v*func_w(x_test))
    I = l1-l
    return(I)

def L_int_0(optimizer_u,x_test,x_test_,y_output_u, optimizer_v, y_output_v,h,t):
    """
        Loss function at interior points at initial step,i.e, t=h
                 (r.f. Eqn. (6) ) 
    """    
    numerator = torch.log(torch.abs(I_0(optimizer_u, x_test,x_test_, y_output_u,optimizer_v,y_output_v,h,t))**2)
    
    denominator = torch.log(torch.sum((y_output_v)**2))
    ratio=numerator-denominator
    
    return(ratio)

def L_int_1(net_u,optimizer_u,x_test,x_test_,y_output_u, optimizer_v,y_output_v,h,t):
    """
        Loss function at interior points after the initial step,i.e, t=ih,i>1
                 (r.f. Eqn. (6) ) 
    """        
    numerator = torch.log(torch.abs(I_1(net_u,optimizer_u,x_test,x_test_,y_output_u,optimizer_v,y_output_v,h,t))**2)
    denominator = torch.log(torch.sum((y_output_v)**2))
    ratio=numerator-denominator
    return(ratio)

def L_bd(net_u,x_boundary, x_boundary_,t):
    """
        Loss function for boundary condition
                (r.f. Eqn.(7) )
    """
    result = torch.mean((net_u(x_boundary_)-func_u(x_boundary,t))**2)
    return(result)


def Loss_u_0(optimizer_u,x_test,x_test_,y_output_u, optimizer_v_0,y_output_v,h,t,net_u,x_boundary, x_boundary_,alpha):
    """
        Loss function for sol at initial step,i.e, t=h
        (r.f. Eqn. (13) without middle term)
    """    
    return(L_int_0(optimizer_u,x_test,x_test_,y_output_u,optimizer_v_0,y_output_v, h,t)\
           +alpha*L_bd(net_u,x_boundary, x_boundary_,t))

def Loss_u_1(optimizer_u,x_test,x_test_,y_output_u, optimizer_v_0,y_output_v,h,t,net_u,x_boundary, x_boundary_,alpha):
    """
        Loss function for sol at all steps after the initial step,i.e, t=ih, i>1
                (r.f. Eqn. (13) without middle term)
    """    
    return(L_int_1(net_u,optimizer_u,x_test,x_test_,y_output_u, optimizer_v_0,y_output_v,h,t)\
           +alpha*L_bd(net_u,x_boundary, x_boundary_,t))

def Loss_v_0(optimizer_u,x_test,x_test_,y_output_u,optimizer_v_0,y_output_v,h,t):
    """
        loss function for test function at initial step,i.e, t=h
    """    
    return(-L_int_0(optimizer_u,x_test,x_test_,y_output_u,optimizer_v_0,y_output_v, h,t))

def Loss_v_1(net_u,optimizer_u,x_test,x_test_,y_output_u, optimizer_v_0,y_output_v,h,t):
    """
        loss function for test function after initial step,i.e, t=ih, i>1
    """
    return(-L_int_1(net_u,optimizer_u,x_test,x_test_,y_output_u, optimizer_v_0,y_output_v,h,t))



   
    
    

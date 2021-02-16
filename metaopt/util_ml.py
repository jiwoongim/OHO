import os, sys
import torch
import numpy as np
from copy import copy, deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from metaopt.util import *
#np.random.seed(1)
#torch.manual_seed(3)


def compute_HessianVectorProd(model, dFdS, data, target, is_cuda=0, logosftmax=1):

    eps_machine = np.finfo(data.data.cpu().numpy().dtype).eps

    ## Compute Hessian Vector product h
    vmax_x, vmax_d = 0, 0
    model_plus = deepcopy(model)
    for param, direction in zip(model_plus.parameters(), dFdS):
        vmax_x = np.maximum(vmax_x, torch.max(torch.abs(param)).item())
        vmax_d = np.maximum(vmax_d, torch.max(abs(direction)).item())
        break

    if vmax_d ==0: vmax_d = 1
    Hess_est_r = np.sqrt(eps_machine) * (1+vmax_x) / vmax_d
    Hess_est_r = max([ Hess_est_r, 0.001])
    for param, direction in zip(model_plus.parameters(), dFdS):
        perturbation =  Hess_est_r * direction
        if is_cuda: perturbation = perturbation.cuda()
        param.data.add_(perturbation)

    for p in model_plus.parameters():
         #print(p.data[0][0])
         break

    model_plus.train()
    output = model_plus(data)
    loss = F.nll_loss(output, target)
    loss.backward()

    model_minus = deepcopy(model)
    for p in model_minus.parameters():
         #print(p.data[0][0])
         break
    
    for param, direction in zip(model_minus.parameters(), dFdS):
        perturbation =  Hess_est_r * direction
        if is_cuda: perturbation = perturbation.cuda()
        param.data.add_(-perturbation)

    for p in model_minus.parameters():
         #print(p.data[0][0])
         break
    
    model_minus.train()
    output = model_minus(data)
    loss = F.nll_loss(output, target)
    loss.backward()

   
    for p in model.parameters():
         #print(p.data[0][0])
         break
    
    g_plus  = [p.grad.data for p in model_plus.parameters()]
    g_minus = [p.grad.data for p in model_minus.parameters()]
    Hv = (flatten_array(g_plus) -
         flatten_array(g_minus)) / (2 * Hess_est_r)

    #import pdb; pdb.set_trace()
    return Hv 



def compute_HessianVectorProd_np(model, dFdS, data, target, is_cuda=0, logosftmax=1):

    eps_machine = np.finfo(data.data.cpu().numpy().dtype).eps

    ## Compute Hessian Vector product h
    vmax_x, vmax_d = 0, 0
    model_plus = deepcopy(model)
    for param, direction in zip(model_plus.parameters(), dFdS):
        vmax_x = np.maximum(vmax_x, torch.max(torch.abs(param)).item())
        vmax_d = np.maximum(vmax_d, np.max(abs(direction)))
         #print(p.data[0][0])
        break

    if vmax_d ==0: vmax_d = 1
    Hess_est_r = np.sqrt(eps_machine) * (1+vmax_x) / vmax_d
    Hess_est_r = max([ Hess_est_r, 0.001])
    for param, direction in zip(model_plus.parameters(), dFdS):
        perturbation = torch.from_numpy(Hess_est_r * direction).type(torch.FloatTensor)
        if is_cuda: perturbation = perturbation.cuda()
        param.data.add_(perturbation)

    for p in model_plus.parameters():
         #print(p.data[0][0])
         break

    model_plus.train()
    output = model_plus(data)
    loss = F.nll_loss(output, target)
    loss.backward()

    model_minus = deepcopy(model)
    for p in model_minus.parameters():
         #print(p.data[0][0])
         break
    
    for param, direction in zip(model_minus.parameters(), dFdS):
        perturbation = torch.from_numpy(Hess_est_r * direction).type(torch.FloatTensor)
        if is_cuda: perturbation = perturbation.cuda()
        param.data.add_(-perturbation)

    for p in model_minus.parameters():
         #print(p.data[0][0])
         break
    
    model_minus.train()
    output = model_minus(data)
    loss = F.nll_loss(output, target)
    loss.backward()

   
    for p in model.parameters():
         #print(p.data[0][0])
         break
    
    g_plus  = [p.grad.data for p in model_plus.parameters()]
    g_minus = [p.grad.data for p in model_minus.parameters()]
    Hv = (flatten_array(g_plus) -
         flatten_array(g_minus)) / (2 * Hess_est_r)

    #import pdb; pdb.set_trace()
    return Hv 


def norm(z):
    """Computes the L2 norm of a numpy array."""
    return torch.sqrt(torch.sum(torch.square(z))).item()


def norm_np(z):
    """Computes the L2 norm of a numpy array."""
    return np.sqrt(np.sum(np.square(z)))


def lr_scheduler_init(optimizer, lrsch_type, gamma=0.95, N=100, step_size=1000):

    if lrsch_type == 'lambda':
        # Assuming optimizer has two groups.
        lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: 0.95 ** epoch
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    elif lrsch_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lrsch_type == 'expstep':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif lrsch_type == 'multstep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25*250,50*250,75*250,100*250], gamma=0.3)

    elif lrsch_type =='cosinestep':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, N, eta_min=0)

    else:
        return None
    return scheduler

def compute_correlation(vec_list, normF=0):
  
    corr_list = []
    N = len(vec_list)
    for i in range(N):
        for j in range(i+1,N):
            vec1 = vec_list[i]
            vec2 = vec_list[j]
            corr = np.dot(vec1,vec2) 
            if not normF: corr = corr / (norm_np(vec1) * norm_np(vec2))
            corr_list.append(corr)
    return np.nanmean(corr_list), np.nanstd(corr_list)




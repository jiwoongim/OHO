import torch, math
import numpy as np
from torch.optim.optimizer import Optimizer
from itertools import tee


class SGD_Multi_LR(Optimizer):

    def __init__(self, params, lr=0.005, weight_decay=0.00001):

        params, params_copy = tee(params)
        LR, WD = [], [] 
        for p in params:
            LR.append(lr*np.ones(p.shape))
            WD.append(weight_decay * np.ones(p.shape))

        defaults = dict(lr=LR, weight_decay=WD)
        super(SGD_Multi_LR, self).__init__(params_copy, defaults)

    def __setstate__(self, state):
        super(SGD_Multi_LR, self).__setstate__(state)


    def step(self):
        """Performs a single optimization step."""

        for group in self.param_groups:
            for param, lr, wd in zip(group['params'], group['lr'], group['weight_decay']):
                if param.grad is None:
                    continue

                d_p = param.grad.data 
                lr = torch.from_numpy(np.asarray([lr]))
                wd = torch.from_numpy(np.asarray([wd]))

                if d_p.is_cuda: 
                    lr = lr.cuda()
                    wd = wd.cuda()

                #if len(param.shape) == 1:
                #    p_change = -lr[0] * d_p
                #else:
                p_change = -lr[0] * (d_p + wd[0] * param.data)
                param.data.add_(p_change)


class SGD_Quotient_LR(Optimizer):

    def __init__(self, params, lr=0.005, weight_decay=0.00001, quotient=2):

        params, params_copy = tee(params)
        LR, WD = [], [] 
        for p in params:
            LR.append(lr*np.ones(p.shape))
            WD.append(weight_decay * np.ones(p.shape))

        self.quotient = quotient
        defaults = dict(lr=LR, weight_decay=WD)
        super(SGD_Quotient_LR, self).__init__(params_copy, defaults)


    def __setstate__(self, state):
        super(SGD_Quotient_LR, self).__setstate__(state)


    def mlp_step(self):
        """Performs a single optimization step."""

        N = len(self.param_groups[0]['params'])
        M = 0
        for param in self.param_groups[0]['params']: 
            if len(param.shape) > 1: 
                M += 1

        freq = M // self.quotient
        lr_list, l2_list = [], []
        quot_i = 0
        for k in range(self.quotient):

            count = 0
            while (count < freq or k == self.quotient-1) and quot_i < N:
                
                param = self.param_groups[0]['params'][quot_i]

                lr_list.append(self.param_groups[0]['lr'][2*k])
                l2_list.append(self.param_groups[0]['weight_decay'][2*k])
                lr_list.append(self.param_groups[0]['lr'][2*k+1])
                l2_list.append(self.param_groups[0]['weight_decay'][2*k+1])
                count += 1
                quot_i += 2
        assert len(lr_list) == N, 'lr length does not match' 
        assert len(l2_list) == N, 'l2 length does not match' 

        for group in self.param_groups:
            for param, lr, wd in zip(group['params'], lr_list, l2_list):
                if param.grad is None:
                    continue
                d_p = param.grad.data 
                lr = torch.from_numpy(np.asarray([lr]))
                wd = torch.from_numpy(np.asarray([wd]))

                if d_p.is_cuda: 
                    lr = lr.cuda()
                    wd = wd.cuda()

                if len(param.shape) == 1:
                    p_change = -lr[0] * d_p
                else:
                    p_change = -lr[0] * (d_p + wd[0] * param.data)
                param.data.add_(p_change)



    def rez_step(self):
        """Performs a single optimization step."""

        N = len(self.param_groups[0]['params'])
        M = 0
        for param in self.param_groups[0]['params']: 
            if len(param.shape) > 1: 
                M += 1

        freq = M // self.quotient
        lr_list, l2_list = [], []

        quot_i = 0
        for k in range(self.quotient):

            count = 0
            while (count <= freq or k == self.quotient-1) and quot_i < N:
                
                param = self.param_groups[0]['params'][quot_i]
                lr_list.append(self.param_groups[0]['lr'][2*k])
                l2_list.append(self.param_groups[0]['weight_decay'][2*k])
                lr_list.append(self.param_groups[0]['lr'][2*k+1])
                l2_list.append(self.param_groups[0]['weight_decay'][2*k+1])
                if len(param.shape) > 2:
                    lr_list.append(self.param_groups[0]['lr'][2*k+1])
                    l2_list.append(self.param_groups[0]['weight_decay'][2*k+1])
                    quot_i += 3
                else:
                    quot_i += 2
                count += 1
    
        assert len(lr_list) == N, 'lr length does not match' 
        assert len(l2_list) == N, 'l2 length does not match' 

        for group in self.param_groups:
            for param, lr, wd in zip(group['params'], lr_list, l2_list):
                if param.grad is None:
                    continue
                d_p = param.grad.data 
                lr = torch.from_numpy(np.asarray([lr]))
                wd = torch.from_numpy(np.asarray([wd]))

                if d_p.is_cuda: 
                    lr = lr.cuda()
                    wd = wd.cuda()
                
                if len(param.shape) == 1:
                    p_change = -lr[0] * d_p
                else:
                    p_change = -lr[0] * (d_p + wd[0] * param.data)
                param.data.add_(p_change)



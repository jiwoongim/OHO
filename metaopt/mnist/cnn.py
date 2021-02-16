"""network.py
Author: @omarschall, 8-20-2019"""

import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from copy import copy
import numpy as np
from metaopt.util_ml import *
from metaopt.util import *


def convOutputSize(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + padding + stride) / stride)  + 1
    return output 

class CNN(nn.Module):

    def __init__(self, atype, cdims, hdims, lr_init, lambda_l2, is_cuda=0, kernel_size=(5,5), stride=1, padding=1, conv_odim=28):
        super(CNN, self).__init__()


        self.n_params = 0 
        self.conv_outdims = []
        for i in range(1, len(cdims)):
            attr = 'clayer_{}'.format(i)
            layer = nn.Conv2d(cdims[i - 1], cdims[i], kernel_size, stride=stride, padding=padding)
            if is_cuda: layer = layer.cuda()
            setattr(self, attr, layer)

            param_size =  cdims[i-1] * cdims[i] * np.prod(kernel_size) + cdims[i] 
            self.n_params += param_size
            conv_odim = convOutputSize(conv_odim, kernel_size[0], stride=stride, padding=padding)
            self.conv_outdims.append(conv_odim)

        hdims = [conv_odim*conv_odim*cdims[-1]] + hdims
        for i in range(1, len(hdims)):
            attr = 'hlayer_{}'.format(i)
            layer = nn.Linear(hdims[i - 1], hdims[-1])
            if is_cuda: layer = layer.cuda()
            setattr(self, attr, layer)
            param_size = (hdims[i - 1] + 1) * hdims[i]
            self.n_params += param_size

        self.atype = atype 
        self.cdims = cdims 
        self.hdims = hdims

        self.param_sizes = [p.numel() for p in self.parameters()]
        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + self.param_sizes)
    
        self.reset_jacob(is_cuda)
        self.eta  = lr_init
        self.lambda_l2 = lambda_l2
        self.name = 'CNN'

    def reset_jacob(self, is_cuda=1):
        self.dFdlr = torch.zeros(self.n_params)
        self.dFdl2 = torch.zeros(self.n_params) 
        self.dFdl2_norm = 0
        self.dFdlr_norm = 0
        if is_cuda:
            self.dFdlr = self.dFdlr.cuda()
            self.dFdl2 = self.dFdl2.cuda()

    def forward(self, x, logsoftmaxF=1):
        for i_layer in range(1, len(self.cdims)):
            attr = 'clayer_{}'.format(i_layer)
            layer = getattr(self, attr)
            x = layer(x)
            if i_layer < len(self.cdims) - 1:

                if self.atype == 'relu':
                    x = F.relu(x)
                else:
                    x = torch.tanh(x)

        N = x.shape[0]
        x = x.view((N,-1))
        for i_layer in range(1, len(self.hdims)):
            attr = 'hlayer_{}'.format(i_layer)
            layer = getattr(self, attr)
            x = layer(x)
            if i_layer < len(self.hdims) - 1: 

                if self.atype == 'relue':
                    x = F.relu(x)
                else:
                    x = torch.tanh(x)
   
        if logsoftmaxF:
            return F.log_softmax(x, dim=1)
        else:
            return F.softmax(x, dim=1)


    def update_dFdlr(self, Hv, param, grad, is_cuda=0, opt_type='sgd', noise=None):

        self.Hlr = self.eta*Hv
        #H = self.Hlr.data.cpu().numpy() if is_cuda else self.Hlr.data.numpy()
        self.Hlr_norm = norm(self.Hlr)
        self.dFdlr_norm = norm(self.dFdlr)
        self.dFdlr.data = self.dFdlr.data * (1-2*self.lambda_l2*self.eta) \
                                - self.Hlr - grad - 2*self.lambda_l2*param
        if opt_type == 'sgld':
            if noise is None: noise = torch.randn(size=param.shape)
            self.dFdlr.data = self.dFdlr.data + 0.5 * torch.sqrt(2*noise / self.eta / N)


    def update_dFdlambda_l2(self, Hv, param, grad, is_cuda=0):
       
        self.Hl2 = self.eta*Hv
        self.Hl2_norm = norm(self.Hl2)
        self.dFdl2_norm = norm(self.dFdl2)
        self.dFdl2.data = self.dFdl2.data * (1-2*self.lambda_l2*self.eta)\
                                            - self.Hl2 - 2*self.eta*param


    def update_eta(self, mlr, val_grad):

        val_grad = flatten_array(val_grad)
        delta =  (val_grad.dot(self.dFdlr)).data.cpu().numpy()
        self.eta -= mlr * delta
        self.eta = np.maximum(0, self.eta)


    def update_lambda(self, mlr, val_grad):

        val_grad = flatten_array(val_grad)
        delta = (val_grad.dot(self.dFdl2))
        self.lambda_l2 -= mlr * delta
        self.lambda_l2 = np.maximum(0, self.lambda_l2)
        self.lambda_l2 = np.minimum(0.0001, self.lambda_l2)


class ACNN(CNN):

    def __init__(self, atype, cdims, hdims, lr_init, lambda_l2, is_cuda=0, kernel_size=(5,5), stride=1, padding=1, conv_odim=28):
        super(CNN, self).__init__()

        self.n_params = 0 
        self.conv_outdims = []
        for i in range(1, len(cdims)):
            attr = 'clayer_{}'.format(i)
            layer = nn.Conv2d(cdims[i - 1], cdims[i], kernel_size, stride=stride, padding=padding)
            if is_cuda: layer = layer.cuda()
            setattr(self, attr, layer)

            param_size =  cdims[i-1] * cdims[i] * np.prod(kernel_size) + cdims[i] 
            self.n_params += param_size
            conv_odim = convOutputSize(conv_odim, kernel_size[0], stride=stride, padding=padding)
            self.conv_outdims.append(conv_odim)

        hdims = [conv_odim*conv_odim*cdims[-1]] + hdims
        for i in range(1, len(hdims)):
            attr = 'hlayer_{}'.format(i)
            layer = nn.Linear(hdims[i - 1], hdims[-1])
            if is_cuda: layer = layer.cuda()
            setattr(self, attr, layer)
            param_size = (hdims[i - 1] + 1) * hdims[i]
            self.n_params += param_size

        self.atype = atype
        self.cdims = cdims 
        self.hdims = hdims

        self.param_sizes = [p.numel() for p in self.parameters()]
        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + self.param_sizes)

        self.reset_jacob(is_cuda)
        self.eta  = np.ones(len(self.param_sizes)) * lr_init
        self.lambda_l2 = np.ones(len(self.param_sizes)) * lambda_l2
        self.name = 'ACNN'


    def _get_adaptive_hyper(self, is_cuda=0):

        layerwise_eta, layerwise_l2, layerwise_eta_np, layerwise_l2_np = [], [], [], []
        for i, shape in enumerate(self.param_shapes):
            layerwise_eta.append(self.eta[i] * torch.ones(shape).flatten())
            layerwise_l2.append(self.lambda_l2[i] * torch.ones(shape).flatten())

        layerwise_l2 = torch.cat(layerwise_l2)
        layerwise_eta = torch.cat(layerwise_eta)

        if is_cuda: 
            layerwise_l2 = layerwise_l2.cuda()
            layerwise_eta = layerwise_eta.cuda()
        return layerwise_eta, layerwise_l2


    def update_dFdlr(self, Hv, param, grad, is_cuda=0, opt_type='sgd', noise=None):

        layerwise_eta, layerwise_l2 = self._get_adaptive_hyper(is_cuda)

        self.Hlr = layerwise_eta*Hv
        self.Hlr_norm = norm(self.Hlr)
        self.dFdlr_norm = norm(self.dFdlr)
        self.dFdlr.data = self.dFdlr.data * (1-2*layerwise_l2*layerwise_eta) \
                                    - self.Hlr - grad - 2*layerwise_l2*param
        if opt_type == 'sgld':
            if noise is None: noise = torch.randn(size=param.shape)
            self.dFdlr.data = self.dFdlr.data + 0.5 * torch.sqrt(2*noise / layerwise_eta_np / N)


    def update_dFdlambda_l2(self, Hv, param, grad, is_cuda=0):

        layerwise_eta, layerwise_l2  = self._get_adaptive_hyper(is_cuda)

        self.Hl2 = layerwise_eta*Hv
        self.Hl2_norm = norm(self.Hl2)
        self.dFdl2_norm = norm(self.dFdl2)
        self.dFdl2.data = self.dFdl2.data * (1-2*layerwise_l2*layerwise_eta)\
                                        - self.Hl2 - 2*layerwise_eta*param


    def update_eta(self, mlr, val_grad):

        dFdlr_ = unflatten_array(self.dFdlr, self.param_cumsum, self.param_shapes)
        for i, (dFdlr_l, val_grad_l) in enumerate(zip(dFdlr_, val_grad)):
            dFdlr_l = flatten_array(dFdlr_l)
            val_grad_l = flatten_array(val_grad_l)
            delta = (val_grad_l.dot(dFdlr_l)).data.cpu().numpy()
            self.eta[i] -= mlr * delta
            self.eta[i] = np.maximum(0, self.eta[i])


    def update_lambda(self, mlr, val_grad):

        dFdl2_ = unflatten_array(self.dFdl2, self.param_cumsum, self.param_shapes)
        for i, (dFdl2_l, val_grad_l) in enumerate(zip(dFdl2_, val_grad)):
            dFdl2_l = flatten_array(dFdl2_l)
            val_grad_l = flatten_array(val_grad_l)
            delta = (val_grad_l.dot(dFdl2_l)).data.cpu().numpy()
            self.lambda_l2[i] -= mlr * delta
            self.lambda_l2[i] = np.maximum(0, self.lambda_l2[i])
            self.lambda_l2[i] = np.minimum(0.0001, self.lambda_l2[i])




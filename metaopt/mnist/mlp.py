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

class MLP_Drop(nn.Module):

    def __init__(self, n_layers, layer_sizes, lr_init, lambda_l2, is_cuda=0):
        super(MLP_Drop, self).__init__()

        self.layer_sizes = layer_sizes
        self.n_layers = n_layers
        self.n_params = 0
        for i in range(1, self.n_layers):
            attr = 'layer_{}'.format(i)
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            if is_cuda: layer = layer.cuda()
            setattr(self, attr, layer)

            param_size = (layer_sizes[i - 1] + 1) * layer_sizes[i]
            self.n_params += param_size

        self.param_sizes = [p.numel() for p in self.parameters()]
        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + self.param_sizes)

        self.reset_jacob(is_cuda)
        self.eta  = lr_init
        self.lambda_l2 = lambda_l2
        self.name = 'MLP'
        self.grad_norm = 0
        self.grad_norm_vl = 0
        self.grad_angle = 0
        self.param_norm = 0
        self.dFdlr_norm = 0
        self.dFdl2_nrom = 0

    def reset_jacob(self, is_cuda=1):
        self.dFdlr = torch.zeros(self.n_params) #np.zeros(self.n_params)
        self.dFdl2 = torch.zeros(self.n_params) #np.zeros(self.n_params)
        self.dFdl2_norm = 0
        self.dFdlr_norm = 0
        if is_cuda: 
            self.dFdlr = self.dFdlr.cuda()
            self.dFdl2 = self.dFdl2.cuda()


    def forward(self, x, logsoftmaxF=1, drop_rate=0.2):

        x = x.view(-1, self.layer_sizes[0])
        for i_layer in range(1, self.n_layers):
            attr = 'layer_{}'.format(i_layer)
            layer = getattr(self, attr)
            x = layer(x)
            if i_layer < self.n_layers - 2:
                #x = F.relu(x)
                x = torch.tanh(x)
                x = nn.functional.dropout(x, p=drop_rate, training=True)
        if logsoftmaxF:
            return F.log_softmax(x, dim=1)
        else:
            return F.softmax(x, dim=1)


    def update_dFdlr(self, Hv, param, grad, is_cuda=0, opt_type='sgd', noise=None, N=50000):

        #grad = flatten_array([p.grad.data.numpy() for p in self.parameters()])
        #tmp = np.ones(self.n_params) * 0.01 
        self.Hlr = self.eta*Hv
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
        self.dFdl2.data = self.dFdl2.data * (1-2*self.lambda_l2*self.eta) \
                                                - self.Hl2  - 2*self.eta*param


    def update_eta(self, mlr, val_grad):

        val_grad = flatten_array(val_grad)
        delta = val_grad.dot(self.dFdlr).data.cpu().numpy()
        self.eta -= mlr * delta
        self.eta = np.maximum(0, self.eta)


    def update_lambda(self, mlr, val_grad):

        val_grad = flatten_array(val_grad)
        delta = val_grad.dot(self.dFdl2).data.cpu().numpy()
        self.lambda_l2 -= mlr * delta 
        self.lambda_l2 = np.maximum(0, self.lambda_l2)
        self.lambda_l2 = np.minimum(0.0001, self.lambda_l2)


class MLP(nn.Module):

    def __init__(self, n_layers, layer_sizes, lr_init, lambda_l2, is_cuda=0):
        super(MLP, self).__init__()


        self.layer_sizes = layer_sizes
        self.n_layers = n_layers
        self.n_params = 0
        for i in range(1, self.n_layers):
            attr = 'layer_{}'.format(i)
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            if is_cuda: layer = layer.cuda()
            setattr(self, attr, layer)

            param_size = (layer_sizes[i - 1] + 1) * layer_sizes[i]
            self.n_params += param_size

        self.param_sizes = [p.numel() for p in self.parameters()]
        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + self.param_sizes)

        self.reset_jacob(is_cuda)
        self.eta  = lr_init
        self.lambda_l2 = lambda_l2
        self.name = 'MLP'
        self.grad_norm = 0
        self.grad_norm_vl = 0
        self.grad_angle = 0
        self.param_norm = 0
        self.dFdlr_norm = 0
        self.dFdl2_nrom = 0

    def reset_jacob(self, is_cuda=1):
        self.dFdlr = torch.zeros(self.n_params) #np.zeros(self.n_params)
        self.dFdl2 = torch.zeros(self.n_params) #np.zeros(self.n_params)
        self.dFdl2_norm = 0
        self.dFdlr_norm = 0
        if is_cuda: 
            self.dFdlr = self.dFdlr.cuda()
            self.dFdl2 = self.dFdl2.cuda()

    def forward(self, x, logsoftmaxF=1):

        x = x.view(-1, self.layer_sizes[0])
        for i_layer in range(1, self.n_layers):
            attr = 'layer_{}'.format(i_layer)
            layer = getattr(self, attr)
            x = layer(x)
            if i_layer < self.n_layers - 1:
                #x = F.relu(x)
                x = torch.tanh(x)
        if logsoftmaxF:
            return F.log_softmax(x, dim=1)
        else:
            return F.softmax(x, dim=1)

    def update_dFdlr(self, Hv, param, grad, is_cuda=0, opt_type='sgd', noise=None, N=50000):

        #grad = flatten_array([p.grad.data.numpy() for p in self.parameters()])
        #tmp = np.ones(self.n_params) * 0.01 
        self.Hlr = self.eta*Hv
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
        self.dFdl2.data = self.dFdl2.data * (1-2*self.lambda_l2*self.eta) \
                                                - self.Hl2  - 2*self.eta*param


    def update_eta(self, mlr, val_grad):

        val_grad = flatten_array(val_grad)
        delta = val_grad.dot(self.dFdlr).data.cpu().numpy()
        self.eta -= mlr * delta
        self.eta = np.maximum(0.0, self.eta)

    def update_lambda(self, mlr, val_grad):

        val_grad = flatten_array(val_grad)
        delta = val_grad.dot(self.dFdl2).data.cpu().numpy()
        self.lambda_l2 -= mlr * delta 
        self.lambda_l2 = np.maximum(0, self.lambda_l2)
        self.lambda_l2 = np.minimum(0.0002, self.lambda_l2)


class AMLP(MLP):

    def __init__(self, n_layers, layer_sizes, lr_init, lambda_l2, is_cuda=0):
        super(MLP, self).__init__()


        self.layer_sizes = layer_sizes
        self.n_layers = n_layers
        self.n_params = 0
        for i in range(1, self.n_layers):
            attr = 'layer_{}'.format(i)
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            if is_cuda: layer = layer.cuda()
            setattr(self, attr, layer)

            param_size = (layer_sizes[i - 1] + 1) * layer_sizes[i]
            self.n_params += param_size

        self.param_sizes = [p.numel() for p in self.parameters()]
        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + self.param_sizes)

        self.reset_jacob(is_cuda)
        self.eta  = np.ones(len(self.param_sizes)) * lr_init
        self.lambda_l2 = np.ones(len(self.param_sizes)) * lambda_l2
        self.name = 'MLP'

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

    def update_dFdlr(self, Hv, param, grad, is_cuda=0, opt_type='sgd', noise=None, N=50000):

        layerwise_eta, layerwise_l2 = self._get_adaptive_hyper(is_cuda)

        self.Hlr = layerwise_eta *Hv
        #H = self.Hlr.data.cpu().numpy() if is_cuda else self.Hlr.data.numpy()
        self.Hlr_norm = norm(self.Hlr)
        self.dFdlr_norm = norm(self.dFdlr)
        self.dFdlr.data = self.dFdlr.data * (1-2*layerwise_l2*layerwise_eta) \
                                - self.Hlr - grad - 2*layerwise_l2*param
        if opt_type == 'sgld':
            if noise is None: noise = torch.randn(size=param.shape)
            self.dFdlr.data = self.dFdlr.data +  0.5 * torch.sqrt(2 * noise  / N / layerwise_eta)


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
            self.lambda_l2[i] = np.minimum(0.0005, self.lambda_l2[i])



class QMLP(MLP):

    def __init__(self, n_layers, layer_sizes, lr_init, lambda_l2, quotient=2, is_cuda=0):
        super(MLP, self).__init__()

        self.layer_sizes = layer_sizes
        self.n_layers = n_layers
        self.n_params = 0
        for i in range(1, self.n_layers):
            attr = 'layer_{}'.format(i)
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            if is_cuda: layer = layer.cuda()
            setattr(self, attr, layer)

            param_size = (layer_sizes[i - 1] + 1) * layer_sizes[i]
            self.n_params += param_size

        self.param_sizes = [p.numel() for p in self.parameters()]
        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + self.param_sizes)

        self.quotient = quotient
        self.reset_jacob(is_cuda)
        self.eta  = np.ones(quotient*2) * lr_init 
        self.lambda_l2 = np.ones(quotient*2) * lambda_l2 
        self.name = 'QMLP'
        self.grad_norm = 0 
        self.grad_norm_vl = 0
        self.grad_angle = 0
        self.param_norm = 0


    def _get_adaptive_hyper(self, is_cuda=0):

        N = len(self.param_shapes)
        freq = N // self.quotient
        layerwise_eta, layerwise_l2, layerwise_eta_np, layerwise_l2_np = [], [], [], []
        for i, shape in enumerate(self.param_shapes):
            quot_i = min(i//freq, self.quotient-1)
            if len(shape) > 1:
                layerwise_eta.append(self.eta[2*quot_i] * torch.ones(shape).flatten())
            else:
                layerwise_eta.append(self.eta[2*quot_i+1] * torch.ones(shape).flatten())
            layerwise_l2.append(self.lambda_l2[quot_i] * torch.ones(shape).flatten())

        layerwise_l2 = torch.cat(layerwise_l2)
        layerwise_eta = torch.cat(layerwise_eta)

        if is_cuda: 
            layerwise_l2 = layerwise_l2.cuda()
            layerwise_eta = layerwise_eta.cuda()
        return layerwise_eta, layerwise_l2


    def update_dFdlr(self, Hv, param, grad, is_cuda=0, opt_type='sgd', noise=None, N=50000):

        layerwise_eta, layerwise_l2 = self._get_adaptive_hyper(is_cuda)

        self.Hlr = layerwise_eta *Hv
        #H = self.Hlr.data.cpu().numpy() if is_cuda else self.Hlr.data.numpy()
        self.Hlr_norm = norm(self.Hlr)
        self.dFdlr_norm = norm(self.dFdlr)
        self.dFdlr.data = self.dFdlr.data * (1-2*layerwise_l2*layerwise_eta) \
                                - self.Hlr - grad - 2*layerwise_l2*param
        if opt_type == 'sgld':
            if noise is None: noise = torch.randn(size=param.shape)
            self.dFdlr.data = self.dFdlr.data +  0.5 * torch.sqrt(2 * noise  / N / layerwise_eta)


    def update_dFdlambda_l2(self, Hv, param, grad, is_cuda=0):

        layerwise_eta, layerwise_l2  = self._get_adaptive_hyper(is_cuda)

        self.Hl2 = layerwise_eta*Hv
        self.Hl2_norm = norm(self.Hl2)
        self.dFdl2_norm = norm(self.dFdl2)
        self.dFdl2.data = self.dFdl2.data * (1-2*layerwise_l2*layerwise_eta)\
                                        - self.Hl2 - 2*layerwise_eta*param


    def update_eta(self, mlr, val_grad):

        quot_i = 0
        N = len(self.param_shapes)
        dFdlr_ = unflatten_array(self.dFdlr, self.param_cumsum, self.param_shapes)
        M = sum([1 for  shape in self.param_shapes if len(shape) > 1])
        freq = M // self.quotient

        for i in range(self.quotient):
            count = 0
            dFdlr_Ws, dFdlr_bs, vgrad_Ws, vgrad_bs = [], [], [], []
            while (count < freq or i==self.quotient-1) and quot_i < len(dFdlr_):

                dFdlr_wi = dFdlr_[quot_i]
                vgrad_wi = val_grad[quot_i]

                dFdlr_bi = dFdlr_[quot_i+1]
                vgrad_bi = val_grad[quot_i+1]

                dFdlr_Ws.append(dFdlr_wi)
                vgrad_Ws.append(vgrad_wi)
                dFdlr_bs.append(dFdlr_bi)
                vgrad_bs.append(vgrad_bi)
                count += 1
                quot_i += 2

            assert len(dFdlr_bs) !=0 , 'Empty bias gradient list'
            assert len(dFdlr_Ws) !=0 , 'Empty weight gradient list'
            dFdlr_l = flatten_array(dFdlr_Ws) 
            val_grad_l = flatten_array(vgrad_Ws)
            delta = (val_grad_l.dot(dFdlr_l)).data.cpu().numpy()
            self.eta[2*i] -= mlr * delta
            self.eta[2*i] = np.maximum(-0.000001, self.eta[2*i])

            #Bias
            assert len(dFdlr_bs) !=0 , 'Empty gradient list'
            dFdlr_lb = flatten_array(dFdlr_bs) 
            val_grad_lb = flatten_array(vgrad_bs)
            delta_b = (val_grad_lb.dot(dFdlr_lb)).data.cpu().numpy()
            self.eta[2*i+1] -= mlr * delta_b
            self.eta[2*i+1] = np.maximum(0, self.eta[2*i+1])

            #dFdlr_l = flatten_array(dFdlr_l)
            #val_grad_l = flatten_array(val_grad_l)
            #delta = (val_grad_l.dot(dFdlr_l)).data.cpu().numpy()
            #self.eta[i] -= mlr * delta
            #self.eta[i] = np.maximum(0, self.eta[i])


    def update_lambda(self, mlr, val_grad):

        quot_i = 0
        N = len(self.param_shapes)
        dFdl2_ = unflatten_array(self.dFdl2, self.param_cumsum, self.param_shapes)
        M = sum([1 for  shape in self.param_shapes if len(shape) > 1])
        freq = M // self.quotient
        for i in range(self.quotient):

            count = 0
            dFdl2_Ws, vgrad_Ws,dFdl2_bs, vgrad_bs  = [], [], [], []

            while (count < freq or i == self.quotient-1) and quot_i < len(dFdl2_):

                dFdl2_wi = dFdl2_[quot_i]
                vgrad_wi = val_grad[quot_i]
                dFdl2_bi = dFdl2_[quot_i+1]
                vgrad_bi = val_grad[quot_i+1]

                #if len(dFdl2_i.shape) > 1:
                dFdl2_Ws.append(dFdl2_wi)
                vgrad_Ws.append(vgrad_wi)
                dFdl2_bs.append(dFdl2_bi)
                vgrad_bs.append(vgrad_bi)
                count += 1
                quot_i += 2

            dFdl2_l = flatten_array(dFdl2_Ws) 
            val_grad_l = flatten_array(vgrad_Ws)
            delta = (val_grad_l.dot(dFdl2_l)).data.cpu().numpy()
            self.lambda_l2[2*i] -= mlr * delta
            self.lambda_l2[2*i] = np.maximum(0, self.lambda_l2[2*i])
            self.lambda_l2[2*i] = np.minimum(0.003, self.lambda_l2[2*i])

            #dFdl2_lb = flatten_array(dFdl2_bs) 
            #val_grad_lb = flatten_array(vgrad_bs)
            #delta_b = (val_grad_lb.dot(dFdl2_lb)).data.cpu().numpy()
            #self.lambda_l2[2*i+1] -= mlr * delta_b
            #self.lambda_l2[2*i+1] = np.maximum(0, self.lambda_l2[2*i+1])
            #self.lambda_l2[2*i+1] = np.minimum(0.0005, self.lambda_l2[2*i+1])



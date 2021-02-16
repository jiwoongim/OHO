'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from metaopt.util_ml import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetDropout(nn.Module):
    def __init__(self, block, num_blocks, \
                num_classes=10, lr_init=0.00001, lambda_l2=0.0, is_cuda=1):
        super(ResNetDropout, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_sizes = [p.numel() for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + self.param_sizes)
        self.n_params = sum(self.param_sizes)

        self.reset_jacob(is_cuda)
        self.eta  = lr_init
        self.lambda_l2 = lambda_l2
        self.name = 'REZ'
        self.grad_norm = 0 
        self.grad_norm_vl = 0
        self.grad_angle = 0
        self.param_norm = 0

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, drop_rate=0.2):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = nn.functional.dropout(out, p=drop_rate, training=True)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.dropout(out, p=drop_rate, training=True)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = nn.functional.dropout(out, p=drop_rate, training=True)
        out = self.linear(out)
        return F.log_softmax(out, dim=1)
        #return out

    def reset_jacob(self, is_cuda):
        self.dFdlr = torch.zeros(self.n_params)
        self.dFdl2 = torch.zeros(self.n_params) 
        self.dFdl2_norm = 0
        self.dFdlr_norm = 0
        if is_cuda:
            self.dFdlr = self.dFdlr.cuda()
            self.dFdl2 = self.dFdl2.cuda()



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, \
                num_classes=10, lr_init=0.00001, lambda_l2=0.0, is_cuda=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_sizes = [p.numel() for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + self.param_sizes)
        self.n_params = sum(self.param_sizes)

        self.reset_jacob(is_cuda)
        self.eta  = lr_init
        self.lambda_l2 = lambda_l2
        self.name = 'REZ'
        self.grad_norm = 0 
        self.grad_norm_vl = 0
        self.grad_angle = 0
        self.param_norm = 0

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=1)
        #return out

    def reset_jacob(self, is_cuda):
        self.dFdlr = torch.zeros(self.n_params)
        self.dFdl2 = torch.zeros(self.n_params) 
        self.dFdl2_norm = 0
        self.dFdlr_norm = 0
        if is_cuda:
            self.dFdlr = self.dFdlr.cuda()
            self.dFdl2 = self.dFdl2.cuda()

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
                                            - self.Hl2 - 2*self.eta*param


    def update_eta(self, mlr, val_grad):

        val_grad = flatten_array(val_grad)
        delta = (val_grad.dot(self.dFdlr)).data.cpu().numpy()
        self.eta -= mlr * delta
        self.eta = np.maximum(0.0, self.eta)
        #self.eta = np.maximum(-0.000001, self.eta)


    def update_lambda(self, mlr, val_grad):

        val_grad = flatten_array(val_grad)
        delta = (val_grad.dot(self.dFdl2)).data.cpu().numpy()
        self.lambda_l2 -= mlr * delta
        self.lambda_l2 = np.maximum(0.0, self.lambda_l2)
        self.lambda_l2 = np.minimum(0.0003, self.lambda_l2)



def ResNet18_Drop(lr_init, lambda_l2):
    return ResNetDropout(BasicBlock, [2, 2, 2, 2], lr_init=lr_init, lambda_l2=lambda_l2)


def ResNet18(lr_init, lambda_l2):
    return ResNet(BasicBlock, [2, 2, 2, 2], lr_init=lr_init, lambda_l2=lambda_l2)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(lr_init, lambda_l2):
    return ResNet(Bottleneck, [3, 4, 6, 3], lr_init=lr_init, lambda_l2=lambda_l2)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])




class AResNet(ResNet):

    def __init__(self, block, num_blocks, \
                num_classes=10, lr_init=0.00001, lambda_l2=0.0, is_cuda=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_sizes = [p.numel() for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + self.param_sizes)
        self.n_params = sum(self.param_sizes)

        self.reset_jacob(is_cuda)
        self.name = 'AREZ'
        self.grad_norm = 0 
        self.grad_norm_vl = 0
        self.grad_angle = 0
        self.param_norm = 0

        self.eta  = np.ones(len(self.param_sizes)) * lr_init
        self.lambda_l2 = np.ones(len(self.param_sizes)) * lambda_l2


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
            self.lambda_l2[i] = np.minimum(0.0002, self.lambda_l2[i])


def AResNet18(lr_init, lambda_l2):
    return AResNet(BasicBlock, [2, 2, 2, 2], lr_init=lr_init, lambda_l2=lambda_l2)



class QResNet(ResNet):

    def __init__(self, block, num_blocks, \
                num_classes=10, lr_init=0.00001, lambda_l2=0.0, is_cuda=1, quotient=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_sizes = [p.numel() for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + self.param_sizes)
        self.n_params = sum(self.param_sizes)

        self.reset_jacob(is_cuda)
        self.name = 'QREZ'
        self.grad_norm = 0 
        self.grad_norm_vl = 0
        self.grad_angle = 0
        self.param_norm = 0

        self.quotient = quotient
        self.eta  = np.ones(quotient*2) * lr_init 
        self.lambda_l2 = np.ones(quotient*2) * lambda_l2 


    def _get_adaptive_hyper(self, is_cuda=0):

        N = len(self.param_shapes)
        freq = N // self.quotient
        #freq = math.ceil(N/self.quotient)
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
            while (count <= freq or i == self.quotient-1) and quot_i < len(dFdlr_):

                #print(i, freq, count, quot_i, len(dFdlr_), len(dFdlr_[quot_i].shape))
                dFdlr_wi = dFdlr_[quot_i]
                vgrad_wi = val_grad[quot_i]
                dFdlr_b1i = dFdlr_[quot_i+1]
                vgrad_b1i = val_grad[quot_i+1]
           
                dFdlr_Ws.append(dFdlr_wi)
                vgrad_Ws.append(vgrad_wi)
                dFdlr_bs.append(dFdlr_b1i)
                vgrad_bs.append(vgrad_b1i)

                if len(dFdlr_wi.shape) == 4:
                    dFdlr_b2i = dFdlr_[quot_i+2]
                    vgrad_b2i = val_grad[quot_i+2]
                    dFdlr_bs.append(dFdlr_b2i)
                    vgrad_bs.append(vgrad_b2i)
                    quot_i += 3
                elif len(dFdlr_wi.shape) == 2:
                    quot_i += 2
                else:
                    import pdb; pdb.set_trace()
                count += 1

                #if len(dFdlr_i.shape) > 1:
                #    dFdlr_Ws.append(dFdlr_i)
                #    vgrad_Ws.append(vgrad_i)
                #    count += 1
                #else:
                #    dFdlr_bs.append(dFdlr_i)
                #    vgrad_bs.append(vgrad_i)
                #print(len(dFdlr_i.shape))
            #print(quot_i, count, len(dFdlr_bs), len(dFdlr_))

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


    def update_lambda(self, mlr, val_grad):

        quot_i = 0
        N = len(self.param_shapes)
        dFdl2_ = unflatten_array(self.dFdl2, self.param_cumsum, self.param_shapes)
        M = sum([1 for  shape in self.param_shapes if len(shape) > 1])
        freq = M // self.quotient
        for i in range(self.quotient):

            count = 0
            dFdl2_Ws, vgrad_Ws,dFdl2_bs, vgrad_bs  = [], [], [], []

            while (count <= freq or i == self.quotient-1) and quot_i < len(dFdl2_):

                dFdl2_wi = dFdl2_[quot_i]
                vgrad_wi = val_grad[quot_i]
                dFdl2_b1i = dFdl2_[quot_i+1]
                vgrad_b1i = val_grad[quot_i+1]
                
                dFdl2_Ws.append(dFdl2_wi)
                vgrad_Ws.append(vgrad_wi)
                dFdl2_bs.append(dFdl2_b1i)
                vgrad_bs.append(vgrad_b1i)

                if len(dFdl2_wi.shape) == 4:
                    dFdl2_b2i = dFdl2_[quot_i+2]
                    vgrad_b2i = val_grad[quot_i+2]
                    dFdl2_bs.append(dFdl2_b2i)
                    vgrad_bs.append(vgrad_b2i)
                    quot_i += 3
                else:
                    quot_i += 2
                count += 1

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
            #self.lambda_l2[2*i+1] = np.minimum(0.0002, self.lambda_l2[2*i+1])


def QResNet18(lr_init, lambda_l2, quotient=2):
    return QResNet(BasicBlock, [2, 2, 2, 2], lr_init=lr_init, lambda_l2=lambda_l2, quotient=quotient)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()



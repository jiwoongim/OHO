import os, sys
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader




"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.num_epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

def flatten_array_np(X):
    """Takes list of arrays in natural shape of the network parameters
    and returns as a flattened 1D numpy array."""

    return np.concatenate([x.flatten() for x in X])


def flatten_array(X):
    """Takes list of arrays in natural shape of the network parameters
    and returns as a flattened 1D numpy array."""

    return torch.cat([x.flatten() for x in X])


def flatten_array_w_0bias_np(X):
    """Takes list of arrays in natural shape of the network parameters
    and returns as a flattened 1D numpy array."""

    vec = []
    for x in X:
        if len(x.shape) == 0 :
            vec.append(np.zeros(x.shape).flatten())
        else:
            vec.append(x.flatten())
    return np.concatenate(vec)


def flatten_array_w_0bias(X):
    """Takes list of arrays in natural shape of the network parameters
    and returns as a flattened 1D numpy array."""

    vec = []
    for x in X:
        if len(x.shape) == 0 :
            vec.append(torch.zeros(x.shape).flatten())
        else:
            vec.append(x.flatten())
    return torch.cat(vec)



def unflatten_array(X, N, param_shapes):
    """Takes flattened array and returns in natural shape for network
    parameters."""
    return [torch.reshape(X[N[i]:N[i + 1]], s) \
        for i, s in enumerate(param_shapes)]



def unflatten_array_np(X, N, param_shapes):
    """Takes flattened array and returns in natural shape for network
    parameters."""
    return [np.reshape(X[N[i]:N[i + 1]], s) \
        for i, s in enumerate(param_shapes)]


def to_torch_variable(data, target=None, is_cuda=False, floatTensorF=0):

    if is_cuda:
        if floatTensorF: 
            data = torch.FloatTensor(data)
        data = data.cuda()

        if target is not None:
            if floatTensorF: 
                target = torch.LongTensor(target)
            target = target.cuda()

    data = Variable(data)
    if target is not None:
        target = Variable(target)
        return data, target

    return data

def get_grads(params, is_cuda=0):

    grads = []
    for p in params:
        if is_cuda:
            grads.append(p.grad.data)
        else:
            grads.append(p.grad.data)

    return grads


def get_grads_np(params, is_cuda=0):

    grads = []
    for p in params:
        if is_cuda:
            grads.append(p.grad.data.cpu().numpy())
        else:
            grads.append(p.grad.data.numpy())

    return grads

def save(model, save_path):
    torch.save([model.state_dict()], save_path+'.pkl')


def load(model, save_path):
    model.load_state_dict(torch.load(save_path+'.pkl', \
                                map_location=lambda storage, \
                                loc: storage)[0])
    return model




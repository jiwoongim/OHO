import os, sys
import numpy as np
import scipy as sp
from itertools import product, cycle

from mlp import * 
from util import *
from main import * 
from visualize import lineplot
basepath = '/misc/vlgscratch4/ChoGroup/imj/'

TRAIN=0
VALID=1
TEST =2


def load_model(args, model_path):

    hdims = [args.xdim] + [args.hdim]*args.num_hlayers + [args.ydim]
    num_layers = args.num_hlayers + 2
    model = MLP(num_layers, hdims, args.lr, args.lambda_l2, is_cuda=0)
    model = load(model, basepath+model_path)
    return model


def main(args, model_path):

    args.batch_size = args.batch_size_vl
    dataset = datasets.MNIST('data/mnist', train=True, download=True,
                                        transform=transforms.Compose(
                                                [transforms.ToTensor()]))
    train_set, valid_set = torch.utils.data.random_split(dataset,[60000 - args.valid_size, args.valid_size])


    data_loader_tr = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    data_loader_vl = DataLoader(valid_set, batch_size=args.batch_size_vl, shuffle=True)
    data_loader_te = DataLoader(datasets.MNIST('data/mnist', train=False, download=True,
                                                        transform=transforms.Compose(
                                                                [transforms.ToTensor()])),
                                                batch_size=args.batch_size, shuffle=True)

    data_loader_tr = cycle(data_loader_tr)
    data_loader_vl = cycle(data_loader_vl)
    dataset = [data_loader_tr, data_loader_vl, data_loader_te]

    data_vl, target_vl = next(dataset[VALID])
    data_tr, target_tr = next(dataset[TRAIN])

    ## Initialize Model and Optimizer
    optimizer = None
    epochs = [0,10,20,30,40,50,60,70]
    N = len(epochs)

    grids, losses, colours, labels, lw, ls = [], [], [], [], [], []
    for i, epoch in enumerate(epochs):
        grid_tr, losses_gradtr_tr, losses_gradtr_vl = get_1Dloss_function(epoch, data_tr, target_tr, data_tr, target_tr, data_vl, target_vl)
        grid_vl, losses_gradvl_tr, losses_gradvl_vl = get_1Dloss_function(epoch, data_vl, target_vl, data_tr, target_tr, data_vl, target_vl)
        grids += [grid_tr, grid_tr]
        losses += [losses_gradtr_tr, losses_gradtr_vl]
        colours += ['indianred', 'tomato']
        lw += [(N-i+1)*.5, (N-i+1)*.5]
        ls += ['-', '--']
        labels += ['Epoch%d' % epoch, None]


        ## Loss surace checking w.r.t Train and Valid gradient direciton
        Xs = [grid_tr, grid_vl, grid_tr, grid_vl]
        Ys = [losses_gradtr_tr, losses_gradvl_tr, losses_gradtr_vl, losses_gradvl_vl] 
        colours_ = ['indianred', 'blue', 'tomato', 'skyblue']
        ls_ = ['-', '-', '--', '--']
        labels_ =['Train Loss Train Grad Dir', 'Train Loss Valid Grad Dir', 'Valid Loss Train Grad Dir', 'Valid Loss Valid Grad Dir']
        fname = 'loss/lossvl_grtr_vs_grvl_lr%f_batchszvl%d_%s_%depoch.png' % (lr, batchsz_vl, opt_type, epoch)
        lineplot(Xs, Ys, colours_, labels_, xlabel='Step size', ylabel='Loss', fname=fname, ls=ls_)



    fname = 'loss/loss1d_function_lr%f_batchszvl%d_%s_Epoch10-60.png' % (lr, batchsz_vl, opt_type)
    lineplot(grids, losses, colours, labels, xlabel='Step size', ylabel='Loss', fname=fname, lw=lw)



def get_1Dloss_function(epoch, data, target, data_tr, target_tr, data_vl, target_vl, num_steps=100, step_size=0.01):

    model = load_model(args, model_path+str(epoch))
    _, _, grads = get_grad_valid(model, data, target, is_cuda)
    grid, model_list = get_grids(model, grads, step_size, num_steps=num_steps)

    losses_tr, losses_vl = [], []
    #for model_tr_i, model_vl_i in zip(model_list_gtr, model_list_gvl):
    for model_i in model_list:

        _, loss_tr, accuracy_tr, _, _ = feval(data_tr, target_tr, model_i, None, mode='eval')
        _, loss_vl, accuracy_vl, _, _ = feval(data_vl, target_vl, model_i, None, mode='eval')
        losses_tr.append(loss_tr)
        losses_vl.append(loss_vl)

    ## Visualize
    fname = 'loss/stepforwad_lr%f_batchszvl%d_%s_%depoch.png' % (lr, batchsz_vl, opt_type, epoch)
    Xs = [grid, grid]
    Ys = [losses_tr, losses_vl] 
    colours = ['indianred', 'tomato']
    labels =['Train Loss', 'Valid Loss']
    ls = ['-', '--']
    lineplot(Xs, Ys, colours, labels, xlabel='Step size', ylabel='Loss', fname=fname, ls=ls)

    return grid, losses_tr, losses_vl


def get_grids(model, directions, step_size, num_steps):

    grid, model_list = [], []
    assert len(model.param_shapes) == len(directions), \
            'number of parameters and directions do not match'

    for step_i in range(1,num_steps+1):
        model_i = deepcopy(model)
        params = model_i.parameters()
        for param,direction in zip(params, directions):
            new_param = param.data - step_size * step_i * direction
            param.data = new_param

        dir_norm = norm(flatten_array_np(directions))
        grid.append(step_i*step_size*dir_norm)
        model_list.append(model_i)

    return grid, model_list



if __name__ == '__main__':

    args = parse_args()
    is_cuda = args.is_cuda

    lr = 0.01
    tot_epoch = 100
    batchsz_vl = 100
    opt_type = 'sgld'
    model_path = '/exp/mlr0.000100_lr0.010000_l20.000010/300epoch_10000vlbz_1updatefreq/checkpoint/epoch'
    model_path = '/exp/mlr0.000100_lr%f_l20.000100/mlp_%depoch_%dvlbz_%s_1updatefreq/checkpoint/epoch' % (lr, tot_epoch, batchsz_vl, opt_type)
    main(args, model_path)




import os, sys, math, argparse, time
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from copy import copy
import numpy as np
import matplotlib.pyplot as plt

from itertools import product, cycle
import pickle

from resnet18 import * 
from metaopt.util import *
from metaopt.util_ml import *
from metaopt.optimizer import SGD_Multi_LR
from metaopt.cifar.main import *

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
ifold=100
RNG = np.random.RandomState(ifold)

TRAIN=0
VALID=1
TEST =2

basepath = '/scratch/ji641'

def sample_hyperparam(args, sampler_type):

    #lr = RNG.rand()
    lr = np.exp(np.random.uniform(-100,0,1))[0]
    lr = lr * (0.2-0.0001) + 0.0001
    args.lr = lr

    #l2 = RNG.rand()
    l2 = np.exp(np.random.uniform(-100,0,1))[0]
    l2 = l2 * (0.0002) 
    args.lambda_l2 = l2

    args.sampler_type = sampler_type
    num_epoch = RNG.randint(low=100, high=300)
    args.num_epoch = num_epoch

    if args.opt_type == 'sgd_step':
        args.step_size = RNG.randint(low=100, high=5000)
    elif args.opt_type == 'sgd_expstep':
        args.gamma = RNG.uniform(0.1,0.99)
    elif args.opt_type == 'adam':
        #lr = RNG.rand()
        lr = np.exp(np.random.uniform(-100,0,1))[0]
        args.lr = lr * (0.003-0.00001) + 0.00001
        args.beta1 = np.exp(RNG.uniform(np.log(0.85),np.log(0.99)))
        args.beta2 = np.exp(RNG.uniform(np.log(0.85),np.log(0.99)))

    return args


def race(args, sampler_type='uniform', thrd=0.):

    dataset = load_cifar10(args)

    print('Model Type: %s Opt Type: %s meta-lr %f lr %f l2 %f, Update Freq %d Reset Freq %d |Nvl| %d' \
            % (args.model_type, args.opt_type, args.mlr,  args.lr, \
                args.lambda_l2, args.update_freq, args.reset_freq, args.valid_size))

    trial, contF = 0, 1
    while contF:
        trial +=1
        args = sample_hyperparam(args, sampler_type)
        if args.opt_type == 'sgd_step':
            print('<<---- Trial %d Lr %f L2 %f Epoch %d step size %d --->>'  % (trial, args.lr, args.lambda_l2, args.num_epoch, args.step_size))
        elif args.opt_type == 'sgd_expstep':
            print('<<---- Trial %d Lr %f L2 %f Epoch %d gamma %f --->>'  % (trial, args.lr, args.lambda_l2, args.num_epoch, args.gamma))
        elif args.opt_type == 'adam':
            print('<<---- Trial %d Lr %f L2 %f Epoch %d beta1 %f beta2 %f --->>'  % (trial, args.lr, args.lambda_l2, args.num_epoch, args.beta1, args.beta2))
        else:
            print('<<---- Trial %d Lr %f L2 %f Epoch %d --->>'  % (trial, args.lr, args.lambda_l2, args.num_epoch))

        te_loss_list = main(args, trial=trial, ifold=ifold)
        ## Model
        #print('==> Building model..')
        ### Initialize Model and Optimizer
        #if args.model_type == 'arez18':
        #    model = AResNet18(args.lr, args.lambda_l2)
        #    optimizer = SGD_Multi_LR(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
        #elif args.model_type == 'rez18':
        #    model = ResNet18(args.lr, args.lambda_l2)
        #    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
        #elif args.model_type == 'rez50':
        #    model = ResNet50(args.lr, args.lambda_l2)
        #    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)

        #if is_cuda:
        #    model = model.to(device)
        #    model = torch.nn.DataParallel(model)
        #    cudnn.benchmark = True

        #os.makedirs('%s/cifar10/trial%d/' % (args.save_dir, trial), exist_ok=True)
        #os.makedirs('%s/cifar10/trial%d/%s/' % (args.save_dir, trial, sampler_type), exist_ok=True)
        #os.makedirs('%s/cifar10/trial%d/%s/mlr%f_lr%f_l2%f/' % (args.save_dir, trial, sampler_type, args.mlr, args.lr, args.lambda_l2), exist_ok=True)
        #fdir = '%s/cifar10/trial%d/%s/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_%dupdatelabmda_fold%d/' \
        #        % (args.save_dir, trial, sampler_type, args.mlr, args.lr, args.lambda_l2, args.model_type, \
        #            args.num_epoch, args.batch_size_vl, args.opt_type, \
        #            args.update_freq, args.reset_freq, args.update_lambda, ifold)
        #os.makedirs(fdir, exist_ok=True)
        #os.makedirs(fdir+'/checkpoint/', exist_ok=True)
        #args.fdir = fdir
        #print(args.fdir)

        ### Train 
        #Wn_list, l2_list, lr_list, dFdlr_list, dFdl2_list, gang_list, tr_epoch, vl_epoch, te_epoch,\
        #                            tr_acc_list, te_acc_list, \
        #                            tr_loss_list, vl_loss_list, te_loss_list \
        #                            = train(args, dataset, model, optimizer, is_cuda=is_cuda)

        #if args.save:
        #    os.makedirs(fdir, exist_ok=True)
        #    np.save(fdir+'Wn', Wn_list)
        #    np.save(fdir+'lr', lr_list)
        #    np.save(fdir+'l2', l2_list)
        #    np.save(fdir+'gang_list', gang_list)
        #    np.save(fdir+'dFdlr_list', dFdlr_list)
        #    np.save(fdir+'dFdl2_list', dFdl2_list)
        #    np.save(fdir+'tr_epoch', tr_epoch)
        #    np.save(fdir+'vl_epoch', vl_epoch)
        #    np.save(fdir+'te_epoch', te_epoch)
        #    np.save(fdir+'tr_loss', tr_loss_list)
        #    np.save(fdir+'vl_loss', vl_loss_list)
        #    np.save(fdir+'te_loss', te_loss_list)
        #    np.save(fdir+'tr_acc', tr_acc_list)
        #    np.save(fdir+'te_acc', te_acc_list)

        print('*** Trial %d Test loss %f Lr %f L2 %f ***' \
                    % (trial, te_loss_list, args.lr, args.lambda_l2))

        #if te_loss_list < thrd:
        #    contF = 0





if __name__ == '__main__':

    args = parse_args()
    is_cuda = 1
    device = 'cuda' if is_cuda else 'cpu'
    race(args)





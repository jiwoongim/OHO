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
ifold=8
RNG = np.random.RandomState(ifold)

TRAIN=0
VALID=1
TEST =2
basepath = '/scratch/ji641'


def ideal_hyper_exp(args, sampler_type=None, thrd=0.):

    dataset = load_cifar10(args)
    trial, contF = 0, 1
    te_loss_list = []
    #quotient_list = [1,2,4,8,16]
    quotient_list = [4]
    for quotient in quotient_list:
        args.quotient = quotient
        args.sampler_type = None
        args.model_type = 'qrez18'
        print('Model Type: %s Opt Type: %s meta-lr %f lr %f l2 %f, Update Freq %d Reset Freq %d |Nvl| %d Quotient %d' \
            % (args.model_type, args.opt_type, args.mlr,  args.lr, \
                args.lambda_l2, args.update_freq, args.reset_freq, args.valid_size, args.quotient))

        trial +=1
        print('<<---- Trial %d Lr %f L2 %f Epoch %d--->>'  % (trial, args.lr, args.lambda_l2, args.num_epoch))
        te_loss = main(args, trial=trial, ifold=ifold, quotient=quotient, device=device)
        te_loss_list.append(te_loss)
        print('*** Trial %d Test loss %f Lr %f L2 %f ***' \
                    % (trial, te_loss_list[-1], args.lr, args.lambda_l2))

    print(quotient_list)
    print(te_loss_list)


if __name__ == '__main__':

    args = parse_args()
    is_cuda = 1
    args.is_cuda = is_cuda
    device = 'cuda' if is_cuda else 'cpu'
    ideal_hyper_exp(args)



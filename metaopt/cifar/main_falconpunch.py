import os, sys, math, argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from copy import copy
import numpy as np
import matplotlib.pyplot as plt

from metaopt.optimizer import SGD_Multi_LR
from itertools import product, cycle
import pickle

from resnet18 import * 
from metaopt.util import *
from metaopt.util_ml import *
from metaopt.cifar.main import meta_update, feval, update_optimizer_hyperparams

TRAIN=0
VALID=1
TEST =2

#torch.manual_seed(3)
RNG = np.random.RandomState(0)


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of DVAE collections"
    parser = argparse.ArgumentParser(description=desc)


    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist'],
                        help='The name of dataset')
    parser.add_argument('--num_epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')
    parser.add_argument('--batch_size_vl', type=int, default=100, help='The size of validation batch')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--model_type', type=str, default='mlp',help="'mlp' | 'amlp'")
    parser.add_argument('--opt_type', type=str, default='sgd', help="'sgd' | 'sgld'")
    parser.add_argument('--shake', type=int, default=50)
    parser.add_argument('--xdim', type=float, default=784)
    parser.add_argument('--hdim', type=float, default=128)
    parser.add_argument('--ydim', type=float, default=10)
    parser.add_argument('--num_hlayers', type=float, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mlr', type=float, default=1e-4)
    parser.add_argument('--lambda_l1', type=float, default=1e-4)
    parser.add_argument('--lambda_l2', type=float, default=1e-4)
    parser.add_argument('--update_lambda', type=int, default=1)
    parser.add_argument('--update_freq', type=int, default=1)
    parser.add_argument('--reset_freq', type=int, default=-0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--valid_size', type=int, default=10000)
    parser.add_argument('--checkpoint_freq', type=int, default=10)
    parser.add_argument('--is_cuda', type=int, default=1)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='/scratch/ji641/imj/')

    return check_args(parser.parse_args())



def load_cifar10(args):

    ## Initialize Dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_set, valid_set = torch.utils.data.random_split(dataset,[50000 - args.valid_size, args.valid_size])

    data_loader_tr = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    data_loader_vl = DataLoader(valid_set, batch_size=args.batch_size_vl, shuffle=True, num_workers=2)
    data_loader_te = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    data_loader_vl = cycle(data_loader_vl)
    dataset = [data_loader_tr, data_loader_vl, data_loader_te]

    return dataset


def main(args, trial=0, ifold=0, device='cuda'):

    dataset = load_cifar10(args)

    # Model
    print('==> Building model..')
    ## Initialize Model and Optimizer
    if args.model_type == 'qrez18':
        model = QResNet18(args.lr, args.lambda_l2, quotient=args.quotient)
        optimizer = SGD_Quotient_LR(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2, quotient=quotient)
    elif args.model_type == 'arez18':
        model = AResNet18(args.lr, args.lambda_l2)
        optimizer = SGD_Multi_LR(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    elif args.model_type == 'rez18':
        model = ResNet18(args.lr, args.lambda_l2)
        if args.opt_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2, betas=(args.beta1,args.beta2))
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
        #optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    elif args.model_type == 'rez18drop':
        model = ResNet18_Drop(args.lr, args.lambda_l2)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    elif args.model_type == 'rez50':
        model = ResNet50(args.lr, args.lambda_l2)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)

    #Update optimizer with new eta
    optimizer = update_optimizer_hyperparams(args, model, optimizer)


    if args.is_cuda:
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True


    os.makedirs('%s/exp/cifar10/shakeL2%d' % (args.save_dir, args.shake), exist_ok=True)
    os.makedirs('%s/exp/cifar10/shakeL2%d/mlr%f_lr%f_l2%f/' % (args.save_dir, args.shake, args.mlr, args.lr, args.lambda_l2), exist_ok=True)
    fdir = '%s/exp/cifar10/shakeL2%d/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_%dupdatelabmda/' \
            % (args.save_dir, args.shake, args.mlr, args.lr, args.lambda_l2, args.model_type, \
                args.num_epoch, args.batch_size_vl, args.opt_type, \
                args.update_freq, args.reset_freq, args.update_lambda)
    
    if args.opt_type == 'sgd_step':
        fdir = fdir.rstrip('/') + '_stepsize%d' % args.step_size
    elif args.opt_type == 'sgd_expstep': 
        fdir = fdir.rstrip('/') + '_gamma%f' % args.gamma

       
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(fdir+'/checkpoint/', exist_ok=True)
    args.fdir = fdir
    print(args.fdir)


    ## Train 
    Wn_list, l2_list, lr_list, dFdlr_list, dFdl2_list, gang_list, tr_epoch, vl_epoch, te_epoch,\
                                tr_acc_list, te_acc_list, \
                                tr_loss_list, vl_loss_list, te_loss_list \
                                = train(args, dataset, model, optimizer)

    if args.save:
        os.makedirs(fdir, exist_ok=True)
        np.save(fdir+'Wn', Wn_list)
        np.save(fdir+'lr', lr_list)
        np.save(fdir+'l2', l2_list)
        np.save(fdir+'gang_list', gang_list)
        np.save(fdir+'dFdlr_list', dFdlr_list)
        np.save(fdir+'dFdl2_list', dFdl2_list)
        np.save(fdir+'tr_epoch', tr_epoch)
        np.save(fdir+'vl_epoch', vl_epoch)
        np.save(fdir+'te_epoch', te_epoch)
        np.save(fdir+'tr_loss', tr_loss_list)
        np.save(fdir+'vl_loss', vl_loss_list)
        np.save(fdir+'te_loss', te_loss_list)
        np.save(fdir+'tr_acc', tr_acc_list)
        np.save(fdir+'te_acc', te_acc_list)


def train(args, dataset, model, optimizer, saveF=0):

    counter = 0
    lr_list, l2_list = [], []
    dFdlr_list, dFdl2_list, Wn_list, gang_list = [], [], [], []
    tr_epoch, tr_loss_list, tr_acc_list = [], [], []
    vl_epoch, vl_loss_list, vl_acc_list = [], [], []
    te_epoch, te_loss_list, te_acc_list = [], [], []
    for epoch in range(args.num_epoch+1):

        if epoch % 10 == 0:
            te_losses, te_accs = [], []
            for batch_idx, (data, target) in enumerate(dataset[TEST]):

                data, target = to_torch_variable(data, target, is_cuda)

                _, loss, accuracy, _, _, _ = feval(data, target, model, optimizer, mode='eval', is_cuda=is_cuda)
                te_losses.append(loss)
                te_accs.append(accuracy)
            te_epoch.append(epoch)
            te_loss_list.append(np.mean(te_losses))
            te_acc_list.append(np.mean(te_accs))
    
            print('Valid Epoch: %d, Loss %f Acc %f' % 
                (epoch, np.mean(te_losses), np.mean(te_accs)))


        for batch_idx, (data, target) in enumerate(dataset[TRAIN]):

            data, target = to_torch_variable(data, target, is_cuda)
            opt_type = args.opt_type
            #if epoch > args.num_epoch * 0.1 and args.opt_type == 'sgld':
            #    opt_type = args.opt_type
            #else:
            #    opt_type = 'sgd'
            model, loss, accuracy, output, noise, _ = feval(data, target, model, optimizer, \
                                is_cuda=is_cuda, mode='meta-train', opt_type=opt_type)
            tr_epoch.append(counter)
            tr_loss_list.append(loss)
            tr_acc_list.append(accuracy)


            if args.reset_freq > 0 and counter % args.reset_freq == 0:
                model.reset_jacob() 

            if counter % args.update_freq == 0 and args.mlr != 0.0:
                data_vl, target_vl = next(dataset[VALID])
                data_vl, target_vl = to_torch_variable(data_vl, target_vl, is_cuda)
                model, loss_vl, optimizer = meta_update(args, data_vl, target_vl, data, target, model, optimizer, noise)
                vl_epoch.append(counter)
                vl_loss_list.append(loss_vl.item())

            counter += 1  
    
        if epoch % args.checkpoint_freq == 0:
            os.makedirs(args.fdir+ '/checkpoint/', exist_ok=True)
            save(model, args.fdir+ '/checkpoint/epoch%d' % epoch) 


        lambda_str = str(model.module.lambda_l2) if args.update_lambda else 'Fixed'
        #if args.update_lambda: lambda_str = str(model_.lambda_l2)  #else 'Fixed'
        model_ = model.module if 'DataParallel' in str(type(model)) else model
        fprint = 'Train Epoch: %d, Tr Loss %f Vl loss %f Acc %f Eta %s, L2 %s, |dFdlr| %.2f |dFdl2| %.2f |G| %.4f |G_vl| %.4f Gang %.3f |W| %.2f'
        print(fprint % (epoch, np.mean(tr_loss_list[-100:]), \
                        np.nanmean(vl_loss_list[-100:]), \
                        np.nanmean(tr_acc_list[-100:]), \
                        str(model_.eta), str(model_.lambda_l2), \
                        model_.dFdlr_norm, model_.dFdl2_norm,\
                        model_.grad_norm,  model_.grad_norm_vl, \
                        model_.grad_angle, model_.param_norm))

        Wn_list.append(model_.param_norm)
        dFdlr_list.append(model_.dFdlr_norm)
        dFdl2_list.append(model_.dFdl2_norm)
        if int(args.num_epoch * args.shake/300) == epoch:
            if 0:
                old_eta = model_.eta
                model_.eta = 0.2 # RNG.rand() * 0.2
                print('--- Switching learning rate from %f %f ---' % (old_eta, model_.eta))
            else :
                old_lambda_l2 = model_.lambda_l2
                model_.lambda_l2 = 0.0 # RNG.rand() * 0.2
                print('--- Switching weight decay coefficient from %f %f ---' % (old_lambda_l2, model_.lambda_l2))

        if args.model_type == 'arez18' or args.model_type == 'qrez18':
            lr_list.append(model_.eta.copy())
            l2_list.append(model_.lambda_l2.copy())
        else:
            lr_list.append(model_.eta)
            l2_list.append(model_.lambda_l2)
        gang_list.append(model_.grad_angle)


    Wn_list = np.asarray(Wn_list)
    l2_list = np.asarray(l2_list)
    lr_list = np.asarray(lr_list)
    dFdlr_list = np.asarray(dFdlr_list)
    dFdl2_list = np.asarray(dFdl2_list)
    tr_epoch = np.asarray(tr_epoch)
    vl_epoch = np.asarray(vl_epoch)
    te_epoch = np.asarray(te_epoch)
    tr_acc_list = np.asarray(tr_acc_list)
    te_acc_list = np.asarray(te_acc_list)
    tr_loss_list = np.asarray(tr_loss_list)
    vl_loss_list = np.asarray(vl_loss_list)
    te_loss_list = np.asarray(te_loss_list)
    gang_list = np.asarray(gang_list)

    return Wn_list, l2_list, lr_list, dFdlr_list, dFdl2_list, gang_list, \
                tr_epoch, vl_epoch, te_epoch, \
                tr_acc_list, te_acc_list, tr_loss_list, vl_loss_list, te_loss_list


if __name__ == '__main__':

    args = parse_args()
    is_cuda = args.is_cuda
    main(args)




import os, sys, math, argparse
from copy import copy
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

from scipy.ndimage.interpolation import rotate
from metaopt.optimizer import SGD_Multi_LR
from itertools import product, cycle
import pickle

from mlp import * 
from metaopt.util import *
from metaopt.util_ml import *
from main import feval, meta_update, get_grad_valid, update_optimizer_hyperparams, load_mnist

TRAIN=0
VALID=1
TEST =2

torch.manual_seed(3)


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
    parser.add_argument('--model_type', type=str, default='mlp',\
                        help="'mlp' | 'amlp'")
    parser.add_argument('--opt_type', type=str, default='sgd', help="'sgd' | 'sgld'")
    parser.add_argument('--ood_type', type=str, default='mixed2', help="'rotation' | 'target'")
    parser.add_argument('--xdim', type=float, default=784)
    parser.add_argument('--hdim', type=float, default=128)
    parser.add_argument('--ydim', type=float, default=10)
    parser.add_argument('--num_hlayers', type=float, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mlr', type=float, default=1e-4)
    parser.add_argument('--lambda_l1', type=float, default=1e-4)
    parser.add_argument('--lambda_l2', type=float, default=1e-4)
    parser.add_argument('--tns_freq', type=int, default=5)
    parser.add_argument('--update_freq', type=int, default=1)
    parser.add_argument('--reset_freq', type=int, default=-0)
    parser.add_argument('--update_lambda', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--valid_size', type=int, default=10000)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--checkpoint_freq', type=int, default=10)
    parser.add_argument('--is_cuda', type=int, default=1)
    parser.add_argument('--save', type=int, default=0)
    #parser.add_argument('--save_dir', type=str, default='/misc/vlgscratch4/ChoGroup/imj/')
    parser.add_argument('--save_dir', type=str, default='/scratch/ji641/imj/')
   
    return check_args(parser.parse_args())


def main(args):

    ## Initialize Dataset
    if args.ood_type == 'imbalance':
        dataset = load_mnist_imbalance(args)
    else:
        dataset = load_mnist(args)

    ## Initialize Model and Optimizer
    hdims = [args.xdim] + [args.hdim]*args.num_hlayers + [args.ydim]
    num_layers = args.num_hlayers + 2
    if args.model_type == 'amlp':
        model = AMLP(num_layers, hdims, args.lr, args.lambda_l2, is_cuda=is_cuda)
        optimizer = SGD_Multi_LR(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    else:
        model = MLP(num_layers, hdims, args.lr, args.lambda_l2, is_cuda=is_cuda)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    print('Model Type: %s Opt Type: %s Update Freq %d Reset Freq %d' \
            % (args.model_type, args.opt_type, args.update_freq, args.reset_freq))

    os.makedirs('%s/exp/mnist/ood/' % args.save_dir, exist_ok=True)
    os.makedirs('%s/exp/mnist/ood/%s/' % (args.save_dir, args.ood_type), exist_ok=True)
    os.makedirs('%s/exp/mnist/ood/%s/mlr%f_lr%f_l2%f/' % (args.save_dir, args.ood_type, args.mlr, args.lr, args.lambda_l2), exist_ok=True)
    if 'mixed' in args.ood_type:
        fdir = '%s/exp/mnist/ood/%s/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq/' \
                % (args.save_dir, args.ood_type, args.mlr, args.lr, args.lambda_l2, \
                    args.model_type, args.num_epoch, args.batch_size_vl, args.opt_type, \
                    args.update_freq, args.reset_freq)
    else:
        fdir = '%s/exp/mnist/ood/%s/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_%dtnsfreq/' \
                % (args.save_dir, args.ood_type, args.mlr, args.lr, args.lambda_l2, \
                    args.model_type, args.num_epoch, args.batch_size_vl, args.opt_type, \
                    args.update_freq, args.reset_freq, args.tns_freq)

    os.makedirs(fdir, exist_ok=True)
    os.makedirs(fdir+'/checkpoint/', exist_ok=True)
    args.fdir = fdir
    print(args.fdir)
    ## Train 
    #Wn_list, l2_list, lr_list, dFdlr_list, dFdl2_list, gang_list, tr_epoch, vl_epoch, te_epoch,\
    #                            tr_acc_list, te_acc_list, \
    #                            tr_loss_list, vl_loss_list, te_loss_list \
    #                            = train(args, dataset, model, optimizer, num_quotient=100)
    if  'mixed' in args.ood_type:
        Wn_list, l2_list, lr_list, dFdlr_list, dFdl2_list, gang_list, tr_epoch, vl_epoch, te_epoch,\
                                tr_acc_list, te_acc_list, \
                                tr_loss_list, vl_loss_list, te_loss_list \
                                = contiuous_manipulation(args, dataset, model, optimizer, saveF=0, visualize=0)
    elif 'imbalance' in args.ood_type:
        Wn_list, l2_list, lr_list, dFdlr_list, dFdl2_list, gang_list, tr_epoch, vl_epoch, te_epoch,\
                                tr_acc_list, te_acc_list, \
                                tr_loss_list, vl_loss_list, te_loss_list \
                                = imbalanced_manipulation(args, dataset, model, optimizer, saveF=0, visualize=0)


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


def load_mnist_imbalance(args):

    ## Initialize Dataset
    dataset = datasets.MNIST('data/mnist', train=True, download=True,
                transform=transforms.Compose([transforms.ToTensor()]))
    
    perm = np.random.permutation(60000)
    tr_data, tr_target = dataset.data[perm][:50000], dataset.targets[perm][:50000]
    vl_data, vl_target = dataset.data[perm][50000:], dataset.targets[perm][50000:]
    #train_set, valid_set = torch.utils.data.random_split(dataset,[60000 - args.valid_size, args.valid_size])
    test_set = datasets.MNIST('data/mnist', train=False, download=True,
                transform=transforms.Compose([transforms.ToTensor()]))

    dataset = [[tr_data, tr_target], [vl_data, vl_target], [test_set.data, test_set.targets]]
    return dataset




def manipulate_imbalance(args, dataset0, epoch, dataset=None):

    ood_scheduler = [(0,0), (1,10), (2,20), (3,30), (4,40), (5,50), (6,60), (7,70), (8,80), (9,90), (0,100)]

    for target_class, breakpoint in ood_scheduler:

        if epoch == breakpoint:
            
            new_dataset = []
            for i in range(3):
                data_i, target_i = dataset0[i][0].data.cpu().numpy(), dataset0[i][1].data.cpu().numpy()
                mask1 = np.argwhere(target_i == target_class).flatten()
                mask0 = np.argwhere(target_i != target_class).flatten()
                perm = np.random.permutation(len(mask0))
                new_mask0 = mask0[perm][:len(mask1)]

                mask = np.hstack([new_mask0, mask1])
                perm = np.random.permutation(len(mask))
                mask = mask[perm] 
                new_data = data_i[mask]
                new_target = target_i[mask]
                new_dataset.append([new_data, new_target])
            return new_dataset
    return dataset

def manipulate_lvl2(args, data, target, epoch, num_quotient=5):

    N,C,D1,D2 = data.shape
    #ood_scheduler = [('normal',5), ('hflip',10), ('rotation',25), ('vflip',30), ('target',50), ('rotation',75), ('normal',100)]
    ood_scheduler = [('normal',10), ('hflip',15), ('normal',20), ('rotation',35), ('normal',45), \
                     ('vflip',50), ('normal',60), ('target',70), ('normal',80), ('rotation',90), ('normal',100)]
    #ood_scheduler = [('normal',30), ('hflip',45), ('normal',60), ('rotation',105), ('normal',135), \
    #                 ('vflip',150), ('normal',180), ('target',210), ('normal',240), ('rotation',270), ('normal',300)]
    #ood_scheduler = [('normal',30), ('hflip',45), ('normal',60), ('rotation',105), ('normal',135), \
    #                 ('vflip',150), ('normal',180), ('target',210), ('normal',240), ('rotation',270), ('normal',300)]

    for ood_type, breakpoint in ood_scheduler:

        if epoch < breakpoint:
            if ood_type == 'target':
                target = (target + epoch) % args.ydim

            elif ood_type == 'rotation':

                angle = (epoch % 10) * 36
                data = data.data.cpu().numpy()
                data = np.asarray([rotate(data[i].squeeze(), angle, reshape=False) for i in range(N)])
                data = data.reshape([N,C,D1,D2])
                data = torch.FloatTensor(data)

            elif ood_type == 'hflip':
                
                data = data.data.cpu().numpy()
                data = np.asarray([np.fliplr(data[i].squeeze()) for i in range(N)])
                data = data.reshape([N,C,D1,D2])
                data = torch.FloatTensor(data)

            elif ood_type == 'vflip':
                
                data = data.data.cpu().numpy()
                data = np.asarray([np.flipud(data[i].squeeze()) for i in range(N)])
                data = data.reshape([N,C,D1,D2])
                data = torch.FloatTensor(data)
    
            elif ood_type == 'normal':
                pass

            else:
                import pdb; pdb.set_trace()
            
            break;

    return data, target



def manipulate(args, data, target, epoch, num_quotient=5):

    freq = args.num_epoch / num_quotient
    if args.ood_type == 'target':
        if epoch > freq and epoch < freq * 2:
            target = (target + 1) % args.ydim
        elif epoch > freq * 2 and epoch < freq * 3:
            target = (target + 2) % args.ydim           
        elif epoch > freq * 3 and epoch < freq * 4:
            target = (target + 3) % args.ydim           

    elif args.ood_type == 'rotate':
        N,C,D1,D2 = data.shape
        angle = 0 
        if epoch > freq and epoch < freq * 2:
            angle=90
        elif epoch > freq * 2 and epoch < freq * 3:
            angle=180
        elif epoch > freq * 3 and epoch < freq * 4:
            angle=270

        if angle > 0:
            data = np.asarray([rotate(data[i].squeeze(), angle, reshape=False) for i in range(N)])
            data = data.reshape([N,C,D1,D2])
            data = torch.FloatTensor(data)
            #if args.is_cuda: data.cuda()

        #plt.figure(figsize=(4, 1))
        #gs = gridspec.GridSpec(1, 12)
        #gs.update(wspace=0, hspace=0)
        #ax = plt.subplot(gs[0])
        #ax.imshow(new_data3.squeeze()[0], cmap='gray');
        #ax.axis('off')
        #ax = plt.subplot(gs[1])
        #ax.imshow(new_data2.squeeze()[0], cmap='gray');
        #ax.axis('off')
        #ax = plt.subplot(gs[2])
        #ax.imshow(new_data1.squeeze()[0], cmap='gray');
        #ax = plt.axis('off');
        #ax = plt.subplot(gs[3])
        #ax.imshow(data.squeeze()[0], cmap='gray');
        #ax = plt.axis('off');
        #plt.savefig('figs/mnist_rotation.png')
    return data, target


def contiuous_manipulation(args, dataset, model, optimizer, saveF=0, visualize=0):

    counter = 0
    lr_list, l2_list = [], []
    dFdlr_list, dFdl2_list, Wn_list, gang_list = [], [], [], []
    tr_epoch, tr_loss_list, tr_acc_list = [], [], []
    vl_epoch, vl_loss_list, vl_acc_list = [], [], []
    te_epoch, te_loss_list, te_acc_list = [], [], []

    ## Optimizer learning rate scheduler
    opt_type = args.opt_type
    if 'step' in opt_type:
        lrsch_type = opt_type.split('_')[-1]
        if 'sgd_expstep' == opt_type:
            scheduler = lr_scheduler_init(optimizer, lrsch_type, gamma=args.gamma)
        elif 'sgd_step' == opt_type:
            print('sgd_step')
            scheduler = lr_scheduler_init(optimizer, lrsch_type, step_size=args.step_size)
        else:
            scheduler = lr_scheduler_init(optimizer, lrsch_type, N=args.num_epoch+1)

    single_data_manipulation = []
    for epoch in range(args.num_epoch):

        ## Test Loss 
        if epoch % 1 == 0:
            te_losses, te_accs = [], []
            for batch_idx, (data, target) in enumerate(dataset[TEST]):

                if  'mixed' in args.ood_type:
                    data, target = manipulate_lvl2(args, data, target, epoch)
                data, target = to_torch_variable(data, target, is_cuda)
                _, loss, accuracy, _, _ = feval(data, target, model, optimizer, mode='eval', is_cuda=is_cuda)
                te_losses.append(loss)
                te_accs.append(accuracy)
            te_epoch.append(epoch)
            te_loss_list.append(np.mean(te_losses))
            te_acc_list.append(np.mean(te_accs))
    
            print('Valid Epoch: %d, Loss %f Acc %f' % 
                (epoch, np.mean(te_losses), np.mean(te_accs)))

            if 'step' in opt_type:           
                scheduler.step()
                model.eta = optimizer.param_groups[0]['lr']

        ## Train Loss 
        for batch_idx, (data, target) in enumerate(dataset[TRAIN]):

            if 'mixed' in args.ood_type:
            #data, target = manipulate(args, data, target, epoch)
                data, target = manipulate_lvl2(args, data, target, epoch)

            data, target = to_torch_variable(data, target, is_cuda)
            if visualize:
                single_data_manipulation.append(data_[2].data.cpu().numpy().squeeze())


            model, loss, accuracy, output, noise = feval(data, target, model, optimizer, \
                                is_cuda=is_cuda, mode='meta-train', opt_type=opt_type)
            tr_epoch.append(counter)
            tr_loss_list.append(loss)
            tr_acc_list.append(accuracy)


            if args.reset_freq > 0 and counter % args.reset_freq == 0:
                model.reset_jacob() 

            if counter % args.update_freq == 0 and args.mlr != 0.0:
                data_vl, target_vl = next(dataset[VALID])
                if  'mixed' in args.ood_type:
                    #data_vl, target_vl = manipulate(args, data_vl, target_vl, epoch)
                    data_vl, target_vl = manipulate_lvl2(args, data_vl, target_vl, epoch)
                data_vl, target_vl = to_torch_variable(data_vl, target_vl, is_cuda)
                model, loss_vl, optimizer = meta_update(args, data_vl, target_vl, data, target, model, optimizer, noise)
                vl_epoch.append(counter)
                vl_loss_list.append(loss_vl.item())

            counter += 1  
    
        if epoch % args.checkpoint_freq == 0:
            os.makedirs(args.fdir+ '/checkpoint/', exist_ok=True)
            save(model, args.fdir+ '/checkpoint/epoch%d' % epoch) 


        fprint = 'Train Epoch: %d, Tr Loss %f Vl loss %f Acc %f Eta %s, L2 %s, |dFdlr| %.2f |dFdl2| %.2f |G| %.4f |G_vl| %.4f Gang %.3f |W| %.2f'
        print(fprint % (epoch, np.mean(tr_loss_list[-100:]), \
                        np.mean(vl_loss_list[-100:]), \
                        np.mean(tr_acc_list[-100:]), \
                        str(model.eta), str(model.lambda_l2), \
                        model.dFdlr_norm, model.dFdl2_norm,\
                        model.grad_norm,  model.grad_norm_vl, \
                        model.grad_angle, model.param_norm))

        Wn_list.append(model.param_norm)
        dFdlr_list.append(model.dFdlr_norm)
        dFdl2_list.append(model.dFdl2_norm)
        if args.model_type == 'amlp':
            lr_list.append(model.eta.copy())
            l2_list.append(model.lambda_l2.copy())
        else:
            lr_list.append(model.eta)
            l2_list.append(model.lambda_l2)
        gang_list.append(model.grad_angle)

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


    if visualize:
        single_data_manipulation = np.asarray(single_data_manipulation)
        N = len(single_data_manipulation) 

        plt.figure()
        gs = gridspec.GridSpec(math.ceil(N/10), 10)

        for i in range(N):

            gs.update(wspace=0, hspace=0)
            ax = plt.subplot(gs[i])
            ax.imshow(single_data_manipulation[i], cmap='gray');
            ax.axis('off')

        plt.savefig('figs/mnist_manipulation.png')


    return Wn_list, l2_list, lr_list, dFdlr_list, dFdl2_list, gang_list, \
                tr_epoch, vl_epoch, te_epoch, \
                tr_acc_list, te_acc_list, tr_loss_list, vl_loss_list, te_loss_list


def imbalanced_manipulation(args, dataset, model, optimizer, saveF=0, visualize=0):

    dataset0 = [[dataset[TRAIN][0].clone(), dataset[TRAIN][1].clone()],
                [dataset[VALID][0].clone(), dataset[VALID][1].clone()],
                [dataset[TEST][0].clone(), dataset[TEST][1].clone()]]

    batch_idx_vl = 0
    counter = 0
    lr_list, l2_list = [], []
    dFdlr_list, dFdl2_list, Wn_list, gang_list = [], [], [], []
    tr_epoch, tr_loss_list, tr_acc_list = [], [], []
    vl_epoch, vl_loss_list, vl_acc_list = [], [], []
    te_epoch, te_loss_list, te_acc_list = [], [], []

    ## Optimizer learning rate scheduler
    opt_type = args.opt_type
    if 'step' in opt_type:
        lrsch_type = opt_type.split('_')[-1]
        if 'sgd_expstep' == opt_type:
            scheduler = lr_scheduler_init(optimizer, lrsch_type, gamma=args.gamma)
        elif 'sgd_step' == opt_type:
            print('sgd_step')
            scheduler = lr_scheduler_init(optimizer, lrsch_type, step_size=args.step_size)
        else:
            scheduler = lr_scheduler_init(optimizer, lrsch_type, N=args.num_epoch+1)

    dataset = None
    single_data_manipulation = []
    for epoch in range(args.num_epoch):

        dataset = manipulate_imbalance(args, dataset0, epoch, dataset)
        num_batch_vl = len(dataset[VALID][0]) // args.batch_size_vl

        ## Test Loss 
        if epoch % 1 == 0:
            te_losses, te_accs = [], []

            num_batch = len(dataset[TEST][0]) // args.batch_size_vl
            for batch_idx in range(num_batch):
                data   = dataset[TEST][0][batch_idx*args.batch_size_vl:(batch_idx+1)*args.batch_size_vl]
                target = dataset[TEST][1][batch_idx*args.batch_size_vl:(batch_idx+1)*args.batch_size_vl]
                data, target = to_torch_variable(data, target, is_cuda, 1)
                _, loss, accuracy, _, _ = feval(data, target, model, optimizer, mode='eval', is_cuda=is_cuda)
                te_losses.append(loss)
                te_accs.append(accuracy)
            te_epoch.append(epoch)
            te_loss_list.append(np.mean(te_losses))
            te_acc_list.append(np.mean(te_accs))
   
            print('Valid Epoch: %d, Loss %f Acc %f' % 
                (epoch, np.mean(te_losses), np.mean(te_accs)))

            if 'step' in opt_type:           
                scheduler.step()
                model.eta = optimizer.param_groups[0]['lr']

        ## Train Loss 
        num_batch_tr = len(dataset[TRAIN][0]) // args.batch_size
        for batch_idx in range(num_batch_tr):
            data   = dataset[TRAIN][0][batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
            target = dataset[TRAIN][1][batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
            data, target = to_torch_variable(data, target, is_cuda, 1)
            if visualize:
                single_data_manipulation.append(data_[2].data.cpu().numpy().squeeze())


            model, loss, accuracy, output, noise = feval(data, target, model, optimizer, \
                                is_cuda=is_cuda, mode='meta-train', opt_type=opt_type)
            tr_epoch.append(counter)
            tr_loss_list.append(loss)
            tr_acc_list.append(accuracy)


            if args.reset_freq > 0 and counter % args.reset_freq == 0:
                model.reset_jacob() 

            if counter % args.update_freq == 0 and args.mlr != 0.0:
                data_vl   = dataset[VALID][0][batch_idx_vl*args.batch_size_vl:(batch_idx_vl+1)*args.batch_size_vl]
                target_vl = dataset[VALID][1][batch_idx_vl*args.batch_size_vl:(batch_idx_vl+1)*args.batch_size_vl]
                data_vl, target_vl = to_torch_variable(data_vl, target_vl, is_cuda, 1)
                batch_idx_vl = (batch_idx_vl + 1) % num_batch_vl

                model, loss_vl, optimizer = meta_update(args, data_vl, target_vl, data, target, model, optimizer, noise)
                vl_epoch.append(counter)
                vl_loss_list.append(loss_vl.item())

            counter += 1  
    
        if epoch % args.checkpoint_freq == 0:
            os.makedirs(args.fdir+ '/checkpoint/', exist_ok=True)
            save(model, args.fdir+ '/checkpoint/epoch%d' % epoch) 


        fprint = 'Train Epoch: %d, Tr Loss %f Vl loss %f Acc %f Eta %s, L2 %s, |dFdlr| %.2f |dFdl2| %.2f |G| %.4f |G_vl| %.4f Gang %.3f |W| %.2f'
        print(fprint % (epoch, np.mean(tr_loss_list[-100:]), \
                        np.mean(vl_loss_list[-100:]), \
                        np.mean(tr_acc_list[-100:]), \
                        str(model.eta), str(model.lambda_l2), \
                        model.dFdlr_norm, model.dFdl2_norm,\
                        model.grad_norm,  model.grad_norm_vl, \
                        model.grad_angle, model.param_norm))

        Wn_list.append(model.param_norm)
        dFdlr_list.append(model.dFdlr_norm)
        dFdl2_list.append(model.dFdl2_norm)
        if args.model_type == 'amlp':
            lr_list.append(model.eta.copy())
            l2_list.append(model.lambda_l2.copy())
        else:
            lr_list.append(model.eta)
            l2_list.append(model.lambda_l2)
        gang_list.append(model.grad_angle)

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


    if visualize:
        single_data_manipulation = np.asarray(single_data_manipulation)
        N = len(single_data_manipulation) 

        plt.figure()
        gs = gridspec.GridSpec(math.ceil(N/10), 10)

        for i in range(N):

            gs.update(wspace=0, hspace=0)
            ax = plt.subplot(gs[i])
            ax.imshow(single_data_manipulation[i], cmap='gray');
            ax.axis('off')

        plt.savefig('figs/mnist_manipulation.png')


    return Wn_list, l2_list, lr_list, dFdlr_list, dFdl2_list, gang_list, \
                tr_epoch, vl_epoch, te_epoch, \
                tr_acc_list, te_acc_list, tr_loss_list, vl_loss_list, te_loss_list



def train(args, dataset, model, optimizer, saveF=0, num_quotient=5):

    counter = 0
    lr_list, l2_list = [], []
    dFdlr_list, dFdl2_list, Wn_list, gang_list = [], [], [], []
    tr_epoch, tr_loss_list, tr_acc_list = [], [], []
    vl_epoch, vl_loss_list, vl_acc_list = [], [], []
    te_epoch, te_loss_list, te_acc_list = [], [], []

    opt_type = args.opt_type
    if 'step' in opt_type:
        lrsch_type = opt_type.split('_')[-1]
        if 'sgd_expstep' == opt_type:
            scheduler = lr_scheduler_init(optimizer, lrsch_type, gamma=args.gamma)
        elif 'sgd_step' == opt_type:
            print('sgd_step')
            scheduler = lr_scheduler_init(optimizer, lrsch_type, step_size=args.step_size)
        else:
            scheduler = lr_scheduler_init(optimizer, lrsch_type, N=args.num_epoch+1)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)


    for epoch in range(args.num_epoch+1):

        if epoch % 1 == 0:
            te_losses, te_accs = [], []
            for batch_idx, (data, target) in enumerate(dataset[TEST]):

                data, target = manipulate(args, data, target, epoch, num_quotient=args.tns_freq)
                data, target = to_torch_variable(data, target, is_cuda)
                _, loss, accuracy, _, _ = feval(data, target, model, optimizer, mode='eval', is_cuda=is_cuda)
                te_losses.append(loss)
                te_accs.append(accuracy)
            te_epoch.append(epoch)
            te_loss_list.append(np.mean(te_losses))
            te_acc_list.append(np.mean(te_accs))
    
            print('Valid Epoch: %d, Loss %f Acc %f' % 
                (epoch, np.mean(te_losses), np.mean(te_accs)))

            if 'step' in opt_type:           
                scheduler.step()
                model.eta = optimizer.param_groups[0]['lr']


        for batch_idx, (data, target) in enumerate(dataset[TRAIN]):

            data, target = manipulate(args, data, target, epoch, num_quotient=args.tns_freq)
            data, target = to_torch_variable(data, target, is_cuda)
            opt_type = args.opt_type
            model, loss, accuracy, output, noise = feval(data, target, model, optimizer, \
                                is_cuda=is_cuda, mode='meta-train', opt_type=opt_type)
            tr_epoch.append(counter)
            tr_loss_list.append(loss)
            tr_acc_list.append(accuracy)


            if args.reset_freq > 0 and counter % args.reset_freq == 0:
                model.reset_jacob() 

            if counter % args.update_freq == 0 and args.mlr != 0.0:
                data_vl, target_vl = next(dataset[VALID])
                data_vl, target_vl = manipulate(args, data_vl, target_vl, epoch, num_quotient=args.tns_freq)
                data_vl, target_vl = to_torch_variable(data_vl, target_vl, is_cuda)
                model, loss_vl, optimizer = meta_update(args, data_vl, target_vl, data, target, model, optimizer, noise)
                vl_epoch.append(counter)
                vl_loss_list.append(loss_vl.item())

            counter += 1  
    
        if epoch % args.checkpoint_freq == 0:
            os.makedirs(args.fdir+ '/checkpoint/', exist_ok=True)
            save(model, args.fdir+ '/checkpoint/epoch%d' % epoch) 


        fprint = 'Train Epoch: %d, Tr Loss %f Vl loss %f Acc %f Eta %s, L2 %s, |dFdlr| %.2f |dFdl2| %.2f |G| %.4f |G_vl| %.4f Gang %.3f |W| %.2f'
        print(fprint % (epoch, np.mean(tr_loss_list[-100:]), \
                        np.mean(vl_loss_list[-100:]), \
                        np.mean(tr_acc_list[-100:]), \
                        str(model.eta), str(model.lambda_l2), \
                        model.dFdlr_norm, model.dFdl2_norm,\
                        model.grad_norm,  model.grad_norm_vl, \
                        model.grad_angle, model.param_norm))

        Wn_list.append(model.param_norm)
        dFdlr_list.append(model.dFdlr_norm)
        dFdl2_list.append(model.dFdl2_norm)
        if args.model_type == 'amlp':
            lr_list.append(model.eta.copy())
            l2_list.append(model.lambda_l2.copy())
        else:
            lr_list.append(model.eta)
            l2_list.append(model.lambda_l2)
        gang_list.append(model.grad_angle)

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
    #contiuous_manipulation(args)



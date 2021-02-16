import os, sys, math, argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from copy import copy
import numpy as np
import matplotlib.pyplot as plt

from metaopt.optimizer import SGD_Multi_LR
from itertools import product, cycle
import pickle

from mlp import * 
from metaopt.util import *
from metaopt.util_ml import *

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
    parser.add_argument('--update_freq', type=int, default=1)
    parser.add_argument('--reset_freq', type=int, default=-0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--valid_size', type=int, default=10000)
    parser.add_argument('--checkpoint_freq', type=int, default=10)
    parser.add_argument('--is_cuda', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='/scratch/ji641/imj/')

    return check_args(parser.parse_args())



def load_mnist(args):

    ## Initialize Dataset
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

    data_loader_vl = cycle(data_loader_vl)
    dataset = [data_loader_tr, data_loader_vl, data_loader_te]
    return dataset


def main(args):

    dataset = load_mnist(args)

    ## Initialize Model and Optimizer
    hdims = [args.xdim] + [args.hdim]*args.num_hlayers + [args.ydim]
    num_layers = args.num_hlayers + 2
    if args.model_type == 'amlp':
        model = AMLP(num_layers, hdims, args.lr, args.lambda_l2, is_cuda=is_cuda)
        optimizer = SGD_Multi_LR(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    elif args.model_type == 'mlp_drop':
        model = MLP_Drop(num_layers, hdims, args.lr, args.lambda_l2, is_cuda=is_cuda)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    else:
        model = MLP(num_layers, hdims, args.lr, args.lambda_l2, is_cuda=is_cuda)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    print('Model Type: %s Opt Type: %s Update Freq %d Reset Freq %d' \
            % (args.model_type, args.opt_type, args.update_freq, args.reset_freq))

    os.makedirs('%s/exp/mnist/shake%d/' % (args.save_dir, args.shake), exist_ok=True)
    os.makedirs('%s/exp/mnist/shake%d/mlr%f_lr%f_l2%f/' % (args.save_dir, args.shake, args.mlr, args.lr, args.lambda_l2), exist_ok=True)
    fdir = '%s/exp/mnist/shake%d/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq/' \
            % (args.save_dir, args.shake, args.mlr, args.lr, args.lambda_l2, args.model_type, args.num_epoch, args.batch_size_vl, args.opt_type, args.update_freq, args.reset_freq)
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

                _, loss, accuracy, _, _ = feval(data, target, model, optimizer, mode='eval', is_cuda=is_cuda)
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
            model, loss, accuracy, output, noise = feval(data, target, model, optimizer, \
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

        if int(args.num_epoch * args.shake/100) == epoch:
            old_eta = model.eta
            model.eta = 0.2 # RNG.rand() * 0.2
            print('--- Switching learning rate from %f %f ---' % (old_eta, model.eta))

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


def feval(data, target, model, optimizer, mode='eval', is_cuda=0, opt_type='sgd', N=50000):

    if mode == 'eval':
        model.eval()
        with torch.no_grad():
            output = model(data)
    else:
        model.train()
        optimizer.zero_grad()
        output = model(data)

    # Compute Loss
    loss = F.nll_loss(output, target)
    pred = output.argmax(dim=1, keepdim=True).flatten()  # get the index of the max log-probability
    accuracy = pred.eq(target).float().mean()

    noise = None
    if 'train' in mode:
        loss.backward()

        if opt_type == 'sgld':
            for i,param in enumerate(model.parameters()):

                noise = torch.randn(size=param.shape)
                if type(model.eta) == type(np.array([])):
                    eps = np.sqrt(model.eta[i]*2/ N) * noise  if model.eta[i] > 0 else 0 * noise
                else:
                    eps = np.sqrt(model.eta*2/ N) * noise  if model.eta > 0 else 0 * noise
                eps = to_torch_variable(eps, is_cuda=is_cuda)
                param.grad.data = param.grad.data + eps.data

        optimizer.step()

    elif 'grad' in mode:
        loss.backward()


    return model, loss.item(), accuracy.item(), output, noise


def meta_update(args, data_vl, target_vl, data_tr, target_tr, model, optimizer, noise=None, is_cuda=1):
   
    #Compute Hessian Vector Product
    param_shapes = model.param_shapes
    dFdlr = unflatten_array(model.dFdlr, model.param_cumsum, param_shapes)
    Hv_lr  = compute_HessianVectorProd(model, dFdlr, data_tr, target_tr, is_cuda=is_cuda)

    dFdl2 = unflatten_array(model.dFdl2, model.param_cumsum, param_shapes)
    Hv_l2  = compute_HessianVectorProd(model, dFdl2, data_tr, target_tr, is_cuda=is_cuda)

    model, loss_valid, grad_valid = get_grad_valid(model, data_vl, target_vl, is_cuda)
    #model, loss_valid, grad_valid = get_grad_valid(model, data_tr, target_tr, is_cuda)

    #Compute angle between tr and vl grad
    grad = flatten_array(get_grads(model.parameters(), is_cuda)).data
    param = flatten_array(model.parameters())#.data.cpu().numpy()
    model.grad_norm = norm(grad)
    model.param_norm = norm(param)
    grad_vl = flatten_array(grad_valid)
    model.grad_angle = torch.dot(grad / model.grad_norm, grad_vl / model.grad_norm_vl).item()

    #Update hyper-parameters   
    model.update_dFdlr(Hv_lr, param, grad, is_cuda, noise=noise)
    model.update_eta(args.mlr, val_grad=grad_valid)
    param = flatten_array_w_0bias(model.parameters()).data
    model.update_dFdlambda_l2(Hv_l2, param, grad, is_cuda)
    model.update_lambda(args.mlr*0.01, val_grad=grad_valid)

    #Update optimizer with new eta
    optimizer = update_optimizer_hyperparams(args, model, optimizer)

    return model, loss_valid, optimizer


def get_grad_valid(model, data, target, is_cuda):

    val_model = deepcopy(model)
    val_model.train()
       
    output = val_model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    grads = get_grads(val_model.parameters(), is_cuda)
    model.grad_norm_vl = norm(flatten_array(grads))
    
    return model, loss, grads


def update_optimizer_hyperparams(args, model, optimizer):

    optimizer.param_groups[0]['lr'] = np.copy(model.eta)
    optimizer.param_groups[0]['weight_decay'] = model.lambda_l2

    return optimizer


if __name__ == '__main__':

    args = parse_args()
    is_cuda = args.is_cuda
    main(args)




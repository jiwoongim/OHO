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
from metaopt.optimizer import SGD_Multi_LR, SGD_Quotient_LR

TRAIN=0
VALID=1
TEST =2

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of DVAE collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--num_epoch', type=int, default=300, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')
    parser.add_argument('--batch_size_vl', type=int, default=1000, help='The size of validation batch')
    parser.add_argument('--save_dir', type=str, default='/misc/vlgscratch4/ChoGroup/imj/',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--model_type', type=str, default='rez18',\
                        help="'rez18' | 'arez18'")
    parser.add_argument('--ood_type', type=str, default='target', help="'rotation' | 'target'")
    parser.add_argument('--opt_type', type=str, default='sgd', help="'sgd' | 'sgld'")
    parser.add_argument('--hdim', type=float, default=128)
    parser.add_argument('--ydim', type=float, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mlr', type=float, default=1e-5)
    parser.add_argument('--lambda_l1', type=float, default=5e-4)
    parser.add_argument('--lambda_l2', type=float, default=5e-4)
    parser.add_argument('--update_lambda', type=int, default=1)
    parser.add_argument('--update_freq', type=int, default=1)
    parser.add_argument('--reset_freq', type=int, default=-0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--step_size', type=int, default=1000)
    parser.add_argument('--valid_size', type=int, default=10000)
    parser.add_argument('--checkpoint_freq', type=int, default=10)
    parser.add_argument('--is_cuda', type=int, default=1)
    parser.add_argument('--ifold', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--sampler_type', type=str, default=None)

    args = parser.parse_args()
    if 'CLUSTER' in os.environ: args.save_dir = '/scratch/ji641/' 
    return check_args(args)


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
    data_loader_gr = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    data_loader_gr = cycle(data_loader_gr)
    dataset = [data_loader_tr, data_loader_vl, data_loader_te, data_loader_gr]

    return dataset


def main(args, trial=0, ifold=0, device='cuda', quotient=None):

    dataset = load_cifar10(args)
    print('Model Type: %s Opt Type: %s meta-lr %f lr %f l2 %f, Update Freq %d Reset Freq %d |Nvl| %d Epoch %d' \
            % (args.model_type, args.opt_type, args.mlr,  args.lr, \
                args.lambda_l2, args.update_freq, args.reset_freq, args.valid_size, args.num_epoch))


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

    if args.sampler_type is None:
        os.makedirs('%s/exp/cifar10/' % args.save_dir, exist_ok=True)
        os.makedirs('%s/exp/cifar10/mlr%f_lr%f_l2%f/' % (args.save_dir, args.mlr, args.lr, args.lambda_l2), exist_ok=True)
        fdir = '%s/exp/cifar10/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_%dupdatelabmda_fold%d/' \
                % (args.save_dir, args.mlr, args.lr, args.lambda_l2, args.model_type, \
                    args.num_epoch, args.batch_size_vl, args.opt_type, \
                    args.update_freq, args.reset_freq, args.update_lambda, ifold)
       
        if args.opt_type == 'sgd_step':
            fdir = fdir.rstrip('/') + '_stepsize%d' % args.step_size
        elif args.opt_type == 'sgd_expstep': 
            fdir = fdir.rstrip('/') + '_gamma%f' % args.gamma

        if quotient is not None:
            fdir = fdir.rstrip('/') + '_quotient%d/' % quotient
        
    else:    
        os.makedirs('%s/cifar10/trial%d/' % (args.save_dir, trial), exist_ok=True)
        os.makedirs('%s/cifar10/trial%d/%s/' % (args.save_dir, trial, args.sampler_type), exist_ok=True)
        os.makedirs('%s/cifar10/trial%d/%s/mlr%f_lr%f_l2%f/' % (args.save_dir, trial, args.sampler_type, args.mlr, args.lr, args.lambda_l2), exist_ok=True)
        fdir = '%s/cifar10/trial%d/%s/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_%dupdatelabmda_fold%d/' \
                % (args.save_dir, trial, args.sampler_type, args.mlr, args.lr, args.lambda_l2, args.model_type, \
                    args.num_epoch, args.batch_size_vl, args.opt_type, \
                    args.update_freq, args.reset_freq, args.update_lambda, ifold)
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(fdir+'/checkpoint/', exist_ok=True)
    args.fdir = fdir
    print(args.fdir)


    ## Train 
    Wn_list, l2_list, lr_list, dFdlr_list, dFdl2_list, gang_list, tr_epoch, vl_epoch, te_epoch,\
                                tr_acc_list, te_acc_list, \
                                tr_loss_list, vl_loss_list, te_loss_list, \
                                tr_corr_mean_list, tr_corr_std_list \
                                = train(args, dataset, model, optimizer, is_cuda=args.is_cuda)

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
        np.save(fdir+'tr_grad_corr_mean', tr_corr_mean_list)
        np.save(fdir+'tr_grad_corr_std', tr_corr_std_list)

    print('Final test loss %f' % te_loss_list[-1])
    print(type(te_loss_list[-1]))
    return te_loss_list[-1]


def train(args, dataset, model, optimizer, saveF=0, is_cuda=1):

    start_time0 = time.time()
    opt_type = args.opt_type
    if 'step' in opt_type:
        lrsch_type = opt_type.split('_')[-1]
        if 'sgd_expstep' == opt_type:
            scheduler = lr_scheduler_init(optimizer, lrsch_type, gamma=args.gamma)
        elif 'sgd_step' == opt_type:
            scheduler = lr_scheduler_init(optimizer, lrsch_type, step_size=args.step_size)
        else:
            scheduler = lr_scheduler_init(optimizer, lrsch_type, N=args.num_epoch+1)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

    counter = 0
    lr_list, l2_list = [], []
    dFdlr_list, dFdl2_list, Wn_list, gang_list = [], [], [], []
    tr_epoch, tr_loss_list, tr_acc_list = [], [], []
    vl_epoch, vl_loss_list, vl_acc_list = [], [], []
    te_epoch, te_loss_list, te_acc_list = [], [], []
    tr_corr_mean_list, tr_corr_std_list = [], []
    optimizer = update_optimizer_hyperparams(args, model, optimizer)

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

            if 'step' in opt_type:           
                scheduler.step()
                model.module.eta = optimizer.param_groups[0]['lr']

        grad_list = []
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(dataset[TRAIN]):

            data, target = to_torch_variable(data, target, is_cuda)
            if 'step' in opt_type:
                model, loss, accuracy, output, noise, grad_vec = feval(data, target, model, optimizer, \
                                is_cuda=is_cuda, mode='train', opt_type=opt_type)
            else:
                model, loss, accuracy, output, noise, grad_vec = feval(data, target, model, optimizer, \
                                is_cuda=is_cuda, mode='meta-train', opt_type=opt_type)
            tr_epoch.append(counter)
            tr_loss_list.append(loss)
            tr_acc_list.append(accuracy)
            if batch_idx % 5 == 0: grad_list.append(grad_vec)

            if args.reset_freq > 0 and counter % args.reset_freq == 0:
                model_ = model.module if 'DataParallel' in str(type(model)) else model
                model_.reset_jacob(is_cuda) 

            if epoch % args.update_freq == 0 and 'step' not in opt_type and args.mlr != 0.0:
                data_vl, target_vl = next(dataset[VALID])
                #data_vl, target_vl = next(dataset[3])
                data_vl, target_vl = to_torch_variable(data_vl, target_vl, is_cuda)

                model, loss_vl, optimizer = meta_update(args, data_vl, target_vl, data, target, model, optimizer, noise, is_cuda=is_cuda)
                vl_epoch.append(counter)
                vl_loss_list.append(loss_vl.item())
            counter += 1  

        #grad_list = np.asarray(grad_list)   
        corr_mean, corr_std = compute_correlation(grad_list, normF=1)
        tr_corr_mean_list.append(corr_mean)
        tr_corr_std_list.append(corr_std)

        end_time = time.time()
        if epoch == 0: print('Single epoch timing %f' % ((end_time-start_time) / 60))

        if epoch % args.checkpoint_freq == 0:
            os.makedirs(args.fdir+ '/checkpoint/', exist_ok=True)
            save(model, args.fdir+ '/checkpoint/epoch%d' % epoch) 

        lambda_str = str(model.module.lambda_l2) if args.update_lambda else 'Fixed'
        #if args.update_lambda: lambda_str = str(model_.lambda_l2)  #else 'Fixed'
        model_ = model.module if 'DataParallel' in str(type(model)) else model
        fprint = 'Train Epoch: %d, Tr Loss %f Vl loss %f Acc %f Eta %s, L2 %s, |dFdlr| %.2f |dFdl2| %.2f |G| %.4f |G_vl| %.4f Gang %.3f |W| %.2f, Grad Corr %f %f'
        print(fprint % (epoch, np.mean(tr_loss_list[-100:]), \
                        np.nanmean(vl_loss_list[-100:]), \
                        np.nanmean(tr_acc_list[-100:]), \
                        str(model_.eta), str(model_.lambda_l2), \
                        model_.dFdlr_norm, model_.dFdl2_norm,\
                        model_.grad_norm,  model_.grad_norm_vl, \
                        model_.grad_angle, model_.param_norm, corr_mean, corr_std))

        Wn_list.append(model_.param_norm)
        dFdlr_list.append(model_.dFdlr_norm)
        dFdl2_list.append(model_.dFdl2_norm)
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
    tr_corr_mean_list = np.asarray(tr_corr_mean_list)
    tr_corr_std_list = np.asarray(tr_corr_std_list)

    end_time0 = time.time()
    print('Total training timing %f' % ((end_time0-start_time0) / 3600))

    return Wn_list, l2_list, lr_list, dFdlr_list, dFdl2_list, gang_list, \
                tr_epoch, vl_epoch, te_epoch, tr_acc_list, te_acc_list, \
                tr_loss_list, vl_loss_list, te_loss_list, tr_corr_mean_list, tr_corr_std_list


def feval(data, target, model, optimizer, mode='eval', is_cuda=1, opt_type='sgd', N=50000):

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
    #criterion = nn.CrossEntropyLoss()
    #loss = criterion(output, target)
    pred = output.argmax(dim=1, keepdim=True).flatten()  # get the index of the max log-probability
    accuracy = pred.eq(target).float().mean()

    grad_vec = []
    noise = None
    if 'train' in mode:
        loss.backward()


        for i,param in enumerate(model.parameters()):
            if opt_type == 'sgld':
                noise = torch.randn(size=param.shape)
                model_ = model.module if 'DataParallel' in str(type(model)) else model
                if type(model_.eta) == type(np.array([])):
                    eps = np.sqrt(model_.eta[i]*2/ N) * noise  if model_.eta[i] > 0 else 0 * noise
                else:
                    eps = np.sqrt(model_.eta*2/ N) * noise  if model_.eta > 0 else 0 * noise
                eps = to_torch_variable(eps, is_cuda=is_cuda)
                param.grad.data = param.grad.data + eps.data
            grad_vec.append(param.grad.data.cpu().numpy().flatten())

        if 'SGD_Quotient_LR' in str(optimizer):
            optimizer.rez_step()
        else:
            optimizer.step()
        grad_vec = np.hstack(grad_vec) 
        grad_vec = grad_vec / norm_np(grad_vec)

    elif 'grad' in mode:
        loss.backward()

    return model, loss.item(), accuracy.item(), output, noise, grad_vec


def meta_update(args, data_vl, target_vl, data_tr, target_tr, model_, optimizer, noise=None, is_cuda=1):

    model = model_.module if 'DataParallel' in str(type(model_)) else model_

    #Compute Hessian Vector Product
    param_shapes = model.param_shapes 
    dFdlr= unflatten_array(model.dFdlr, model.param_cumsum, param_shapes)
    Hv_lr  = compute_HessianVectorProd(model, dFdlr, data_tr, target_tr, is_cuda=is_cuda)

    if args.update_lambda:
        dFdl2 = unflatten_array(model.dFdl2, model.param_cumsum, param_shapes)
        Hv_l2  = compute_HessianVectorProd(model, dFdl2, data_tr, target_tr, is_cuda=is_cuda)

    model, loss_valid, grad_valid = get_grad_valid(model, data_vl, target_vl, is_cuda)
    #model, loss_valid, grad_valid = get_grad_valid(model, data_tr, target_tr, is_cuda)

    #Compute angle between tr and vl grad
    grad = flatten_array(get_grads(model.parameters(), is_cuda))
    param = flatten_array(model.parameters())
    model.grad_norm = norm(grad)
    model.param_norm = norm(param)
    grad_vl = flatten_array(grad_valid)
    model.grad_angle = torch.dot(grad / model.grad_norm, grad_vl / model.grad_norm_vl).item()


    #Update hyper-parameters
    model.update_dFdlr(Hv_lr, param, grad, is_cuda, noise=noise)
    model.update_eta(args.mlr, val_grad=grad_valid)
    param = flatten_array_w_0bias(model.parameters()).data
    if args.update_lambda:
        model.update_dFdlambda_l2(Hv_l2, param, grad, is_cuda)
        model.update_lambda(args.mlr, val_grad=grad_valid)

    #Update optimizer with new eta
    optimizer = update_optimizer_hyperparams(args, model, optimizer)

    return model_, loss_valid, optimizer


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

    model_ = model.module if 'DataParallel' in str(type(model)) else model
    optimizer.param_groups[0]['lr'] = np.copy(model_.eta)
    optimizer.param_groups[0]['weight_decay'] = model_.lambda_l2

    return optimizer



if __name__ == '__main__':

    args = parse_args()
    is_cuda = args.is_cuda
    device = 'cuda' if is_cuda else 'cpu'

    torch.manual_seed(args.ifold) 
    main(args, ifold=args.ifold)





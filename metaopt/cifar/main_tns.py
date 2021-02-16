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
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

from scipy.ndimage.interpolation import rotate
from itertools import product, cycle
import pickle

from resnet18 import * 
from metaopt.util import *
from metaopt.util_ml import *
from metaopt.optimizer import SGD_Multi_LR, SGD_Quotient_LR
from metaopt.cifar.main import feval, meta_update, get_grad_valid, update_optimizer_hyperparams, load_cifar10, parse_args

TRAIN=0
VALID=1
TEST =2

torch.manual_seed(5)
#torch.manual_seed(0)


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
        os.makedirs('%s/exp/cifar10/ood/' % args.save_dir, exist_ok=True)
        os.makedirs('%s/exp/cifar10/ood/%s/' % (args.save_dir, args.ood_type), exist_ok=True)
        os.makedirs('%s/exp/cifar10/ood/%s/mlr%f_lr%f_l2%f/' % (args.save_dir, args.ood_type, args.mlr, args.lr, args.lambda_l2), exist_ok=True)
        fdir = '%s/exp/cifar10/ood/%s/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_%dupdatelabmda_fold%d/' \
                % (args.save_dir, args.ood_type, args.mlr, args.lr, args.lambda_l2, args.model_type, \
                    args.num_epoch, args.batch_size_vl, args.opt_type, \
                    args.update_freq, args.reset_freq, args.update_lambda, ifold)
       
        if args.opt_type == 'sgd_step':
            fdir = fdir.rstrip('/') + '_stepsize%d/' % args.step_size
        elif args.opt_type == 'sgd_expstep': 
            fdir = fdir.rstrip('/') + '_gamma%f/' % args.gamma

        if quotient is not None:
            fdir = fdir.rstrip('/') + '_quotient%d/' % quotient
        
    else:    
        os.makedirs('%s/cifar10/trial%d/' % (args.save_dir, trial), exist_ok=True)
        os.makedirs('%s/cifar10/trial%d/%s/' % (args.save_dir, trial, args.sampler_type), exist_ok=True)
        os.makedirs('%s/cifar10/trial%d/%s/mlr%f_lr%f_l2%f/' % (args.save_dir, trial, args.sampler_type, args.mlr, args.lr, args.lambda_l2), exist_ok=True)
        fdir = '%s/cifar10/trial%d/%s/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_%dupdatelabmda_fold%d' \
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
                                tr_loss_list, vl_loss_list, te_loss_list \
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

    print('Final test loss %f' % te_loss_list[-1])
    print(type(te_loss_list[-1]))
    return te_loss_list[-1]


def manipulate(args, data, target, epoch, num_quotient=5, ood_type=None):

    if ood_type is None: ood_type = args.ood_type 

    freq = args.num_epoch / num_quotient
    if ood_type == 'target':
        if epoch > freq and epoch < freq * 2:
            target = (target + 1) % args.ydim
        elif epoch > freq * 2 and epoch < freq * 3:
            target = (target + 2) % args.ydim           
        elif epoch > freq * 3 and epoch < freq * 4:
            target = (target + 3) % args.ydim           

    elif ood_type == 'hflip':
        import pdb; pdb.set_trace()

    elif ood_type == 'rotate':
        N,C,D1,D2 = data.shape
        angle = 0 
        if epoch > freq and epoch < freq * 2:
            angle=90
        elif epoch > freq * 2 and epoch < freq * 3:
            angle=180
        elif epoch > freq * 3 and epoch < freq * 4:
            angle=270

        if angle > 0:
            data = data.permute(0,2,3,1)
            data = np.asarray([rotate(data[i].squeeze(), angle, reshape=False) for i in range(N)])
            data = data.reshape([N,D1,D2,C]).transpose(0,3,1,2)
            data = torch.FloatTensor(data)
            #if args.is_cuda: data.cuda()

        #new_data3 = np.asarray([rotate(data[0].squeeze(), 90, reshape=False)])
        #new_data2 = np.asarray([rotate(data[0].squeeze(), 180, reshape=False)])
        #new_data1 = np.asarray([rotate(data[0].squeeze(), 270, reshape=False)])
        #plt.figure(figsize=(4, 1))
        #gs = gridspec.GridSpec(1, 12)
        #gs.update(wspace=0, hspace=0)
        #ax = plt.subplot(gs[0])
        #ax.imshow(new_data3.squeeze().transpose(1,2,0),interpolation='bicubic');
        #ax.axis('off')
        #ax = plt.subplot(gs[1])
        #ax.imshow(new_data2.squeeze().transpose(1,2,0),interpolation='bicubic');
        #ax.axis('off')
        #ax = plt.subplot(gs[2])
        #ax.imshow(new_data1.squeeze().transpose(1,2,0),interpolation='bicubic');
        #ax = plt.axis('off');
        #ax = plt.subplot(gs[3])
        #
        #ax.imshow(data[0].data.cpu().numpy().transpose(1,2,0),interpolation='bicubic');
        #ax = plt.axis('off');
        #plt.savefig('figs/cifar_rotation.png')
        #import pdb; pdb.set_trace() 
    return data, target


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
    for epoch in range(args.num_epoch+1):

        if epoch % 10 == 0:
            te_losses, te_accs = [], []
            for batch_idx, (data, target) in enumerate(dataset[TEST]):

                data, target = manipulate(args, data, target, epoch, num_quotient=5)
                data, target = to_torch_variable(data, target, is_cuda)
                _, loss, accuracy, _, _ = feval(data, target, model, optimizer, mode='eval', is_cuda=is_cuda)
                te_losses.append(loss)
                te_accs.append(accuracy)
            te_epoch.append(epoch)
            te_loss_list.append(np.mean(te_losses))
            te_acc_list.append(np.mean(te_accs))
    
            print('Valid Epoch: %d, Loss %f Acc %f' % 
                (epoch, np.mean(te_losses), np.mean(te_accs)))

            if '_step' in opt_type:           
                scheduler.step()
                model.module.eta = optimizer.param_groups[0]['lr']

        if 'expstep' in opt_type:           
            scheduler.step()
            model.module.eta = optimizer.param_groups[0]['lr']

        start_time = time.time()
        for batch_idx, (data, target) in enumerate(dataset[TRAIN]):

            data, target = manipulate(args, data, target, epoch, num_quotient=5)
            data, target = to_torch_variable(data, target, is_cuda)
            if 'step' in opt_type:
                model, loss, accuracy, output, noise = feval(data, target, model, optimizer, \
                                is_cuda=is_cuda, mode='train', opt_type=opt_type)
            else:
                model, loss, accuracy, output, noise = feval(data, target, model, optimizer, \
                                is_cuda=is_cuda, mode='meta-train', opt_type=opt_type)
            tr_epoch.append(counter)
            tr_loss_list.append(loss)
            tr_acc_list.append(accuracy)


            if args.reset_freq > 0 and counter % args.reset_freq == 0:
                model_ = model.module if 'DataParallel' in str(type(model)) else model
                model_.reset_jacob(is_cuda) 

            if counter % args.update_freq == 0 and 'step' not in opt_type and args.mlr != 0.0:
                data_vl, target_vl = next(dataset[VALID])
                data_vl, target_vl = manipulate(args, data_vl, target_vl, epoch, num_quotient=5)
                data_vl, target_vl = to_torch_variable(data_vl, target_vl, is_cuda)
                model, loss_vl, optimizer = meta_update(args, data_vl, target_vl, data, target, model, optimizer, noise, is_cuda=is_cuda)
                vl_epoch.append(counter)
                vl_loss_list.append(loss_vl.item())
            counter += 1  

        end_time = time.time()
        if epoch == 0: print('Single epoch timing %f' % ((end_time-start_time) / 60))

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

    end_time0 = time.time()
    print('Total training timing %f' % ((end_time0-start_time0) / 3600))

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
    #criterion = nn.CrossEntropyLoss()
    #loss = criterion(output, target)
    pred = output.argmax(dim=1, keepdim=True).flatten()  # get the index of the max log-probability
    accuracy = pred.eq(target).float().mean()

    noise = None
    if 'train' in mode:
        loss.backward()

        if opt_type == 'sgld':
            for i,param in enumerate(model.parameters()):

                noise = torch.randn(size=param.shape)
                model_ = model.module if 'DataParallel' in str(type(model)) else model
                if type(model_.eta) == type(np.array([])):
                    eps = np.sqrt(model_.eta[i]*2/ N) * noise  if model_.eta[i] > 0 else 0 * noise
                else:
                    eps = np.sqrt(model_.eta*2/ N) * noise  if model_.eta > 0 else 0 * noise
                eps = to_torch_variable(eps, is_cuda=is_cuda)
                param.grad.data = param.grad.data + eps.data

        optimizer.step()

    elif 'grad' in mode:
        loss.backward()


    return model, loss.item(), accuracy.item(), output, noise




if __name__ == '__main__':

    args = parse_args()
    is_cuda = args.is_cuda
    device = 'cuda' if is_cuda else 'cpu'

   
    main(args)





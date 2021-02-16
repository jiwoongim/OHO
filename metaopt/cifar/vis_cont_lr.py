import os, sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from metaopt.visualize import *
plt.rcParams.update({'font.size': 32})
basepath = '/misc/vlgscratch4/ChoGroup/imj/'


def continous_learning_results(model_paths, labels, ood_type, fname):

    fig, axs = plt.subplots(2, 2, figsize=(16, 16), sharey=False)
    ax00 = axs[0,0]
    ax00.set_xlabel('Updates')
    ax00.set_ylabel('Train Loss')

    ax01 = axs[0,1]
    ax01.set_xlabel('Epoch')
    ax01.set_ylabel('Test Loss')

    ax10 = axs[1,0]
    ax10.set_xlabel('Epoch')
    ax10.set_ylabel('Learning Rate')

    ax11 = axs[1,1]
    ax11.set_xlabel('Epoch')
    ax11.set_ylabel('L2 Weight Decay')

    colours = ['tomato', 'skyblue', 'mediumpurple', 'green']
    tr_loss_list, te_loss_list, lr_list, l2_list = [], [], [], []
    for i, (model_path, label) in enumerate(zip(model_paths, labels)):

        fdir = basepath + model_path
        tr_epoch = np.load(fdir+'tr_epoch.npy')
        te_epoch = np.load(fdir+'te_epoch.npy')

        tr_loss = np.load(fdir+'tr_loss.npy')
        te_loss = np.load(fdir+'te_loss.npy')
        lr_ = np.load(fdir+'lr.npy')
        l2_ = np.load(fdir+'l2.npy')

        ax00.plot(tr_epoch, tr_loss, color=colours[i], label=label, alpha=0.5, lw=5)
        ax01.plot(te_epoch, te_loss, color=colours[i], label=label, alpha=0.5, lw=5)
        ax10.plot(np.arange(len(lr_)), lr_, color=colours[i], label=label, alpha=0.5, lw=5)
        ax11.plot(np.arange(len(l2_)), l2_, color=colours[i], label=label, alpha=0.5, lw=5)

    ax00.legend()
    ax01.legend() 
    ax01.legend() 
    ax11.legend() 
    ax00.set_yscale('log')
    ax01.set_yscale('log')

    plt.suptitle('Hyper-paramter Update Frequency Analysis')
    plt.tight_layout()
    plt.savefig(fname, Format='png')
    plt.close()


if __name__ == '__main__':

    ood_type = 'rotate'
    basepath = '/scratch/ji641/'
    labels = ['SGD EXP Decay', 'Meta-Opt']
    fname = './figs/cifar/ood/%s/learning_comparison_update_freq_%s_%s.png' % (ood_type, 'mlp', 'sgd')   
    model_paths = [
        "/exp/cifar10/ood/%s/mlr0.000000_lr0.100000_l20.000000/rez18_300epoch_1000vlbz_sgd_expstep_1updatefreq_0resetfreq_1updatelabmda_fold0_gamma0.970000/" % ood_type,
        "/exp/cifar10/ood/%s/mlr0.000010_lr0.100000_l20.000000/rez18_300epoch_1000vlbz_sgd_1updatefreq_0resetfreq_1updatelabmda_fold0/" % ood_type]    
    #continous_learning_results(model_paths, labels, ood_type, fname)

    ood_type = 'target'
    fname = './figs/cifar/ood/%s/learning_comparison_update_freq_%s_%s.png' % (ood_type, 'mlp', 'sgd')   
    model_paths = [
        "/exp/cifar10/ood/%s/mlr0.000000_lr0.100000_l20.000000/rez18_300epoch_1000vlbz_sgd_expstep_1updatefreq_0resetfreq_1updatelabmda_fold0_gamma0.970000/" % ood_type,
        "/exp/cifar10/ood/%s/mlr0.000010_lr0.100000_l20.000000/rez18_300epoch_1000vlbz_sgd_1updatefreq_0resetfreq_1updatelabmda_fold0/" % ood_type]    
    #continous_learning_results(model_paths, labels, ood_type, fname)


    sequential_manipulation()


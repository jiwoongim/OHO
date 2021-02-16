import os, sys
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from metaopt.visualize import *
matplotlib.rcParams.update({'font.size': 32})
plt.rcParams.update({'font.size': 32})
basepath = '/misc/vlgscratch4/ChoGroup/imj/'


def continous_learning_results(model_paths, labels, ood_type, fname, \
                    xlabels=['Update', 'Epoch', 'Epoch', 'Epoch'],\
                    ylabels=['Train Loss', 'Test Loss', 'Learning Rate', 'L2 Weight Decay'],
                    ftag=['_tr', '_te', '_lr', '_l2']):

    #fig1, axs00 = plt.subplots(1, 1, figsize=(16, 16), sharey=False)
    #ax00.set_xlabel('Updates')
    #ax00.set_ylabel('Train Loss')

    #ax01.set_xlabel('Epoch')
    #ax01.set_ylabel('Test Loss')

    #ax10 = axs[2]
    #ax10.set_xlabel('Epoch')
    #ax10.set_ylabel('Learning Rate')

    #ax11 = axs[3]
    #ax11.set_xlabel('Epoch')
    #ax11.set_ylabel('L2 Weight Decay')

    tr_loss_list, te_loss_list, lr_list, l2_list = [], [], [], []
    for i in range(4): 
        fig1 =plt.figure(figsize=(16, 16))
        ax00 = plt.subplot(1,1,1)
        ax00.set_xlabel(xlabels[i])
        ax00.set_ylabel(ylabels[i])
        colours = ['tomato', 'skyblue', 'mediumpurple', 'green']

        for j, (model_path, label) in enumerate(zip(model_paths, labels)):
            fdir = basepath + model_path
            tr_epoch = np.load(fdir+'tr_epoch.npy')
            te_epoch = np.load(fdir+'te_epoch.npy')

            tr_loss = np.load(fdir+'tr_loss.npy')
            te_loss = np.load(fdir+'te_loss.npy')
            lr_ = np.load(fdir+'lr.npy')
            l2_ = np.load(fdir+'l2.npy')

            if i==0:ax00.plot(tr_epoch, tr_loss, color=colours[j], label=label, alpha=0.5, lw=5)
            if i==1:ax00.plot(te_epoch, te_loss, color=colours[j], label=label, alpha=0.5, lw=10)
            if i==2:ax00.plot(np.arange(len(lr_)), lr_, color=colours[j], label=label, alpha=0.5, lw=10)
            if i==3:ax00.plot(np.arange(len(l2_)), l2_, color=colours[j], label=label, alpha=0.5, lw=10)

        if i == 0: ax00.legend()
        if i <= 1: ax00.set_yscale('log')
        #ax01.set_yscale('log')
        #plt.suptitle('Hyper-paramter Update Frequency Analysis')
        plt.tight_layout()
        plt.savefig(fname+ftag[i]+'.pdf')#, format='png')
        plt.close()
    
  
if __name__ == '__main__':

    ood_type = 'rotate'
    basepath = '/scratch/ji641/'
    labels = ['Fixed SGD', 'Meta-Opt']
    model_paths = [
        "/exp/mnist/ood/%s/mlr0.000000_lr0.100000_l20.000100/mlp_100epoch_100vlbz_sgd_expstep_1updatefreq_0resetfreq/" % ood_type,
        "/exp/mnist/ood/%s/mlr0.000010_lr0.001000_l20.000100/mlp_100epoch_100vlbz_sgd_1updatefreq_0resetfreq/" % ood_type]    
    #continous_learning_results(model_paths, labels, ood_type, fname)

    ood_type = 'target'
    model_paths = [
        "/exp/mnist/ood/%s/mlr0.000000_lr0.100000_l20.000100/mlp_100epoch_100vlbz_sgd_expstep_1updatefreq_0resetfreq/" % ood_type,
        "/exp/mnist/ood/%s/mlr0.000010_lr0.001000_l20.000100/mlp_100epoch_100vlbz_sgd_1updatefreq_0resetfreq/" % ood_type]    
    #continous_learning_results(model_paths, labels, ood_type, fname)


    ood_type='sequential'
    basepath = '/scratch/ji641/'
    labels = ['SGD Fixed', 'SGD Exp', 'Meta-Opt']
    fname = './figs/mnist/ood/%s_%s' % (ood_type, 'mlp')   
    model_paths = [
            'imj/exp/mnist/ood/mixed2/mlr0.000000_lr0.100000_l20.000000/mlp_100epoch_1000vlbz_sgd_1updatefreq_0resetfreq/',
            'imj/exp/mnist/ood/mixed2/mlr0.000000_lr0.100000_l20.000000/mlp_100epoch_1000vlbz_sgd_expstep_1updatefreq_0resetfreq/',
            'imj/exp/mnist/ood/mixed2/mlr0.000010_lr0.100000_l20.000000/mlp_100epoch_1000vlbz_sgd_1updatefreq_0resetfreq/'
                 ]
    #continous_learning_results(model_paths, labels, ood_type, fname) 

    ood_type='sequential'
    basepath = '/scratch/ji641/'
    labels = ['SGD Fixed', 'SGD Exp', 'Meta-Opt']
    fname = './figs/mnist/ood/%s_%s_v2' % (ood_type, 'mlp')   
    model_paths = [
            'imj/exp/mnist/ood/mixed2/mlr0.000000_lr0.100000_l20.000000/mlp_100epoch_1000vlbz_sgd_1updatefreq_0resetfreq/',
            'imj/exp/mnist/ood/mixed2/mlr0.000000_lr0.100000_l20.000000/mlp_100epoch_1000vlbz_sgd_expstep_1updatefreq_0resetfreq/',
            'imj/exp/mnist/ood/mixed2/mlr0.000010_lr0.010000_l20.000000/mlp_100epoch_1000vlbz_sgd_1updatefreq_0resetfreq/'
                 ]
    continous_learning_results(model_paths, labels, ood_type, fname) 



    ood_type='sequential'
    basepath = '/scratch/ji641/'
    labels = ['SGD Fixed', 'SGD Exp', 'Meta-Opt']
    fname = './figs/mnist/ood/%s_%s_mixed3' % (ood_type, 'mlp')   
    model_paths = [
            'imj/exp/mnist/ood/mixed3/mlr0.000000_lr0.100000_l20.000000/mlp_300epoch_100vlbz_sgd_1updatefreq_0resetfreq/',
            'imj/exp/mnist/ood/mixed3/mlr0.000000_lr0.100000_l20.000000/mlp_300epoch_100vlbz_sgd_expstep_1updatefreq_0resetfreq/',
            'imj/exp/mnist/ood/mixed3/mlr0.000010_lr0.100000_l20.000000/mlp_300epoch_100vlbz_sgd_1updatefreq_0resetfreq/'
                 ]
    #continous_learning_results(model_paths, labels, ood_type, fname) 

    ood_type='sequential'
    basepath = '/scratch/ji641/'
    labels = ['SGD Fixed', 'SGD Exp', 'Meta-Opt']
    fname = './figs/mnist/ood/%s_%s_mixed3_v2' % (ood_type, 'mlp')   
    model_paths = [
            'imj/exp/mnist/ood/mixed3/mlr0.000000_lr0.100000_l20.000000/mlp_300epoch_100vlbz_sgd_1updatefreq_0resetfreq/',
            'imj/exp/mnist/ood/mixed3/mlr0.000000_lr0.100000_l20.000000/mlp_300epoch_100vlbz_sgd_expstep_1updatefreq_0resetfreq/',
            'imj/exp/mnist/ood/mixed3/mlr0.000010_lr0.001000_l20.000000/mlp_300epoch_100vlbz_sgd_1updatefreq_0resetfreq/'
                 ]
    #continous_learning_results(model_paths, labels, ood_type, fname) 






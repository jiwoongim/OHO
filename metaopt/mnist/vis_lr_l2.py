import os, sys
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from metaopt.visualize import *
matplotlib.rcParams.update({'font.size':32})
plt.rcParams.update({'font.size': 32})
basepath = '/misc/vlgscratch4/ChoGroup/imj/'

def gradient_correlation_analyis(lr, l2, model_type, num_epoch, batch_vl, opt_type):

    basepath = '/scratch/ji641/'
    fig1, ax1 = plt.subplots(figsize=(16, 16))
    ax1.set_xlabel('Correlation Average ')
    ax1.set_ylabel('Correlation Standard Deviation')

    labels = ['SGD (fixed lr=0.1)', 'OMO']
    colours = ['tomato', 'skyblue', 'mediumpurple', 'green']
    tr_loss_list, te_loss_list, lr_list, l2_list = [], [], [], []
    for i, mlr in enumerate([0.0, 0.00001]):

        fdir = basepath +'/imj/exp/mnist/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq/' \
            % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq, 0)
        label = labels[i]
        tr_corr_mean_list= np.load(fdir+'tr_grad_corr_mean.npy')
        tr_corr_std_list = np.load(fdir+'tr_grad_corr_std.npy')
        ax1.plot(tr_corr_mean_list, tr_corr_std_list, color=colours[i], label=label, alpha=0.5, lw=3, marker='.')
        ax1.scatter(tr_corr_mean_list[0], tr_corr_std_list[0], color=colours[i], alpha=0.85, s=7, marker='*')
        ax1.annotate('Start', xy=(tr_corr_mean_list[0], tr_corr_std_list[0]), xytext=(0, 3),  # 3 points vertical offset
                                                color=colours[i], textcoords="offset points",ha='center', va='bottom')

    ax1.legend() 
    #ax1.set_xscale('log')

    plt.tight_layout()
    plt.savefig('./figs/mnist/gradient_correlation_%s_%s.pdf' % (model_type, opt_type), format='pdf')
    plt.close()



def mo_shake_plot():

    basepath = '/scratch/ji641/'
    fig1, ax00 = plt.subplots(figsize=(16, 16))
    fig2, ax01 = plt.subplots(figsize=(16, 16))
    fig3, ax10 = plt.subplots(figsize=(16, 16))
    fig4, ax11 = plt.subplots(figsize=(16, 16))

    ax00.set_xlabel('Updates')
    ax00.set_ylabel('Train Loss')
    ax01.set_xlabel('Epoch')
    ax01.set_ylabel('Test Loss')
    ax10.set_xlabel('Epoch')
    ax10.set_ylabel('Learning Rate')
    ax11.set_xlabel('Epoch')
    ax11.set_ylabel('L2 Weight Decay')

    colours = ['tomato', 'orange', 'limegreen', 'skyblue', 'mediumpurple', 'gray']
    tr_loss_list, te_loss_list, lr_list, l2_list = [], [], [], []
    #for i, update_freq in enumerate([1, 10, 100]):
    for i, shake in enumerate([10, 30, 50]):

        fdir = basepath +'/imj/exp/mnist/shake%d/mlr0.000005_lr0.001000_l20.000000/cnn_100epoch_100vlbz_sgd_1updatefreq_0resetfreq/' % shake
        #fdir = basepath +'/imj/exp/mnist/shakeD%d/mlr0.000010_lr0.001000_l20.000000/cnn_100epoch_100vlbz_sgd_1updatefreq_0resetfreq/' % shake
        label = 'Epoch %d' % shake 
        tr_epoch = np.load(fdir+'tr_epoch.npy')
        te_epoch = np.load(fdir+'te_epoch.npy')

        tr_loss = np.load(fdir+'tr_loss.npy')
        te_loss = np.load(fdir+'te_loss.npy')
        lr_ = np.load(fdir+'lr.npy')
        l2_ = np.load(fdir+'l2.npy')

        ax00.plot(tr_epoch, tr_loss, color=colours[i], label=label, alpha=0.75, lw=2)
        ax01.plot(te_epoch, te_loss, color=colours[i], label=label, alpha=0.75, lw=3)
        ax10.plot(np.arange(len(lr_)), lr_, color=colours[i], label=label, alpha=0.75, lw=3)
        ax11.plot(np.arange(len(l2_)), l2_, color=colours[i], label=label, alpha=0.75, lw=3)


        fdir = basepath +'/imj/exp/mnist/shake%d/mlr0.000000_lr0.100000_l20.000000/cnn_100epoch_100vlbz_sgd_1updatefreq_0resetfreq/' % shake
        tr_epoch = np.load(fdir+'tr_epoch.npy')
        te_epoch = np.load(fdir+'te_epoch.npy')

        tr_loss = np.load(fdir+'tr_loss.npy')
        te_loss = np.load(fdir+'te_loss.npy')
        lr_ = np.load(fdir+'lr.npy')
        l2_ = np.load(fdir+'l2.npy')

        ax00.plot(tr_epoch, tr_loss, color=colours[i], label=label, alpha=0.5, lw=2, ls='--')
        ax01.plot(te_epoch, te_loss, color=colours[i], label=label, alpha=0.5, lw=3, ls='--')
        ax10.plot(np.arange(len(lr_)), lr_, color=colours[i], label=label, alpha=0.5, lw=3, ls='--')
        ax11.plot(np.arange(len(l2_)), l2_, color=colours[i], label=label, alpha=0.5, lw=3, ls='--')

    ax00.legend()
    ax01.legend() 
    ax00.set_yscale('log')
    ax01.set_yscale('log')

    #plt.suptitle('Learning Rate Change')
    fig1.tight_layout()
    fig1.savefig('./figs/mnist/random_lrChnage_tr.pdf', format='pdf')
    fig2.tight_layout()
    fig2.savefig('./figs/mnist/random_lrChnage_te.pdf', format='pdf')
    fig3.tight_layout()
    fig3.savefig('./figs/mnist/random_lrChnage_lr.pdf', format='pdf')

    plt.close()


def mnist_varying_batchvl():

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
    for i, update_freq in enumerate([1, 10, 100]):

        'rez18_300epoch_500vlbz_sgd_1updatefreq_0resetfreq'
        fdir = basepath +'/exp/mnist/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq/' \
            % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq, 0)
        label = 'Update every %d' % update_freq
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
    plt.savefig('./figs/mnist/learning_comparison_update_freq_%s_%s.png' % (model_type, opt_type), format='pdf')
    plt.close()



def mnist_varying_update_freq(mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type):

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
    for i, update_freq in enumerate([1, 10, 100]):

        fdir = basepath +'/exp/mnist/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq/' \
            % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq, 0)
        label = 'Update every %d' % update_freq
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
    plt.savefig('./figs/mnist/learning_comparison_update_freq_%s_%s.png' % (model_type, opt_type), format='pdf')
    plt.close()



def mnist_varying_jacob(mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq):

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
    for i, reset_freq in enumerate([0, 1, 100, 500]):
        if reset_freq == 0:
            fdir = basepath +'/exp/mnist/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_0resetfreq/' \
                % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq)
            label = 'No reset'
        else:
            fdir = basepath +'/exp/mnist/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq/' \
                % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq, reset_freq)
            label = 'Reset every %d' % reset_freq
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

    plt.suptitle('Jacobian (dTheta/dHyper) Reset Analysis')
    plt.tight_layout()
    plt.savefig('./figs/mnist/learning_comparison_reset_freq_%s_%s.png' % (model_type, opt_type), format='png')
    plt.close()



def main_single_lr(mlr, lr, l2, num_epoch, batch_sz, batch_vl, model_type, opt_type, update_freq):

    fdir = basepath +'/exp/mnist/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq/' % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq)
    lr_list1 = np.load(fdir+'lr.npy')
    l2_list1 = np.load(fdir+'l2.npy')
    dFlr_list1 = np.load(fdir+'dFdlr_list.npy')
    dFl2_list1 = np.load(fdir+'dFdl2_list.npy')
    te_epoch1 = np.load(fdir+'te_epoch.npy')
    tr_data_list1 = np.load(fdir+'tr_loss.npy')
    te_data_list1 = np.load(fdir+'te_loss.npy')

    mlr = 0.0
    fdir = basepath +'/exp/mnist/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq/' % (mlr, lr, l2, model_type, num_epoch, batch_vl, 'sgd', update_freq)
    #fdir = basepath+'/exp/mnist/mlr%f_lr%f_l2%f/%depoch_%dvlbz/' % (mlr, lr, l2, num_epoch, batch_vl)
    lr_list2 = np.load(fdir+'lr.npy')
    l2_list2 = np.load(fdir+'l2.npy')
    dFlr_list2 = np.load(fdir+'dFdlr_list.npy')
    dFl2_list2 = np.load(fdir+'dFdl2_list.npy')
    te_epoch2 = np.load(fdir+'te_epoch.npy')
    tr_data_list2 = np.load(fdir+'tr_loss.npy')
    te_data_list2 = np.load(fdir+'te_loss.npy')


    updates = np.arange(tr_data_list1.shape[0])
    epochs = np.arange(num_epoch+1) * len(updates) / (num_epoch+1)
    te_epoch1 = te_epoch1* len(updates) / num_epoch
    te_epoch2 = te_epoch2* len(updates) / num_epoch
    #te_data_list1 = np.mean(te_data_list1.reshape([batch_sz, -1]), axis=0)
    #te_data_list2 = np.mean(te_data_list2.reshape([batch_sz, -1]), axis=0)

    color='indianred'
    fig, axs = plt.subplots(2, 2, figsize=(16, 16), sharey=False)
    ax00 = axs[0,0]
    ax00.set_xlabel('Updates')
    ax00.set_ylabel('Loss', color=color)
    ax00.plot(updates, tr_data_list1, color='indianred', label='Online Train', alpha=0.5)
    ax00.plot(te_epoch1, te_data_list1, color='indianred', ls='--', label='Online Test')
    ax00.plot(updates, tr_data_list2, color='green', label='Fixed Train', alpha=0.5)
    ax00.plot(te_epoch2, te_data_list2, color='green', ls='--', label='Fixed Test')
    ax00.legend()

    ax01 = axs[0,1]
    ax01.set_xlabel('Updates')
    ax01.set_ylabel('|dPdLr|', color=color)
    ax01.plot(epochs, dFlr_list2, color='green', ls='-', label='Fixed')
    ax01.plot(epochs, dFlr_list1, color='indianred', ls='-', label='Online')
    ax01.legend() 

    ax01 = axs[1,0]
    ax01.set_xlabel('Updates')
    ax01.set_ylabel('|dPdL2|', color=color)
    ax01.plot(epochs, dFl2_list2, color='green', ls='-', label='Fixed')
    ax01.plot(epochs, dFl2_list1, color='indianred', ls='-', label='Online')
    ax01.legend() 


    color='mediumpurple'
    ax01 = axs[1,1]
    ax01.set_ylabel('L2 Weight Decay', color=color)
    ax01.plot(epochs, l2_list1, color=color, lw=3)
    ax01.grid(True)

    color='indianred'
    ax01a = ax01.twinx()
    ax01a.set_ylabel('Learning Rate', color=color)
    ax01a.plot(epochs, lr_list1, color=color, lw=3)
    ax01a.grid(True)

    plt.tight_layout()
    plt.savefig('./figs/mnist/learning_comparison_%s_%s.pdf' % (model_type, opt_type), format='pdf')
    plt.close()


def main_layerwise_hyperparam(model_path, opt_type):

    lr_list = np.load(basepath+model_path+'lr.npy').T
    l2_list = np.load(basepath+model_path+'l2.npy').T
    Xs = [np.arange(len(lr_list.T))]*8

    colours=['red', 'tomato', 'blue','skyblue','black', 'gray', 'green', 'limegreen']
    labels = ['W1', 'B1', 'W2', 'B2', 'W3', 'B3', 'W4', 'B4']
    ls = ['-','--','-','--','-','--','-','--']

    fname = 'amlp/%s_0.01lr_0.0001mlr_100batchszvl_lr.png' % opt_type
    xlabel = 'Epoch'
    ylabel = 'Learning Rate'
    lineplot(Xs, lr_list, colours, labels, xlabel, ylabel, fname, ls=ls, lw=3)

    xlabel = 'Epoch'
    ylabel = 'L2 Coef.'
    fname = 'amlp/%s_0.01lr_0.0001mlr_100batchszvl_l2.png' % opt_type
    lineplot(Xs[::2], l2_list[::2], colours[::2], labels[::2], xlabel, ylabel, fname, ls='-', lw=3)



def main_mlp_loss_comp(update_freq):

    opt_type = 'sgd'
    model_path = "/exp/mnist/mlr0.000000_lr0.010000_l20.000100/mlp_100epoch_100vlbz_%s_%dupdatefreq/" % (opt_type, 1)
    te_epoch0 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss0 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch0 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss0 = np.load(basepath + model_path+'tr_loss.npy')

    opt_type = 'sgd'
    model_path = "/exp/mnist/mlr0.000100_lr0.001000_l20.000100/mlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq/" % (opt_type, update_freq)
    te_epoch1 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss1 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch1 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss1 = np.load(basepath + model_path+'tr_loss.npy')

    opt_type = 'sgld'
    model_path = "/exp/mnist/mlr0.000100_lr0.001000_l20.000100/mlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq/" % (opt_type, update_freq)
    te_epoch2 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss2 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch2 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss2 = np.load(basepath + model_path+'tr_loss.npy')
    
    opt_type = 'sgd'
    model_path = "/exp/mnist/mlr0.000100_lr0.001000_l20.000100/amlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq/" % (opt_type, update_freq)
    te_epoch3 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss3 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch3 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss3 = np.load(basepath + model_path+'tr_loss.npy')

    opt_type = 'sgld'
    model_path = "/exp/mnist/mlr0.000100_lr0.001000_l20.000100/amlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq/" % (opt_type, update_freq)
    te_epoch4 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss4 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch4 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss4 = np.load(basepath + model_path+'tr_loss.npy')

    #Xs = [te_epoch0, te_epoch1, te_epoch2, te_epoch4]
    #Vs = [tr_epoch0, tr_epoch1, tr_epoch2, tr_epoch4]
    #Ys = [te_loss0, te_loss1, te_loss2, te_loss4] 
    #Zs = [tr_loss0, tr_loss1, tr_loss2, tr_loss4]  
    #labels_ =['Fixed SGD', 'Online SGD', 'Online SGLD', 'Online SGLD per layer']
    Xs = [te_epoch0, te_epoch1, te_epoch2, te_epoch3, te_epoch4]
    Vs = [tr_epoch0, tr_epoch1, tr_epoch2, tr_epoch3, tr_epoch4]
    Ys = [te_loss0, te_loss1, te_loss2, te_loss3, te_loss4] 
    Zs = [tr_loss0, tr_loss1, tr_loss2, tr_loss3, tr_loss4] 
    labels_ =['Fixed SGD', 'Online SGD', 'Online SGLD', 'Online SGD per layer', 'Online SGLD per layer']
    colours_ = ['indianred', 'mediumpurple', 'skyblue', 'tomato', 'cyan']
    ls_ = ['-', '-', '-', '-', '-']
    fname = 'loss/loss_te_curve_comp_%s.png'  % update_freq
    lineplot(Xs, Ys, colours_, labels_, xlabel='Epoch', ylabel='Loss', fname=fname, ls=ls_, lw=3, logyscale=1)

    fname = 'loss/loss_tr_curve_comp_%s.png' % update_freq
    lineplot(Vs, Zs, colours_, labels_, xlabel='Updates', ylabel='Loss', fname=fname, ls=ls_, lw=3, logyscale=1)


def main_cnn_loss_comp():

    opt_type = 'sgd'
    model_path = "/exp/mnist/mlr0.000000_lr0.001000_l20.000010/cnn_100epoch_1000vlbz_%s_10updatefreq/" % opt_type
    te_epoch0 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss0 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch0 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss0 = np.load(basepath + model_path+'tr_loss.npy')

    opt_type = 'sgd'
    model_path = "/exp/mnist/mlr0.000004_lr0.001000_l20.000010/cnn_100epoch_1000vlbz_%s_1updatefreq/" % opt_type
    te_epoch1 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss1 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch1 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss1 = np.load(basepath + model_path+'tr_loss.npy')

    opt_type = 'sgld'
    model_path = "/exp/mnist/mlr0.000004_lr0.001000_l20.000010/cnn_100epoch_1000vlbz_%s_1updatefreq/" % opt_type
    te_epoch2 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss2 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch2 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss2 = np.load(basepath + model_path+'tr_loss.npy')
    
    #opt_type = 'sgd'
    #model_path = "/exp/mnist/mlr0.000100_lr0.010000_l20.000100/acnn_100epoch_100vlbz_%s_1updatefreq/" % opt_type
    #te_epoch3 = np.load(basepath + model_path+'te_epoch.npy')
    #te_loss3 = np.load(basepath + model_path+'te_loss.npy')
    #tr_epoch3 = np.load(basepath + model_path+'tr_epoch.npy')
    #tr_loss3 = np.load(basepath + model_path+'tr_loss.npy')

    #opt_type = 'sgld'
    #model_path = "/exp/mnist/mlr0.000100_lr0.010000_l20.000100/acnn_100epoch_100vlbz_%s_1updatefreq/" % opt_type
    #te_epoch4 = np.load(basepath + model_path+'te_epoch.npy')
    #te_loss4 = np.load(basepath + model_path+'te_loss.npy')
    #tr_epoch4 = np.load(basepath + model_path+'tr_epoch.npy')
    #tr_loss4 = np.load(basepath + model_path+'tr_loss.npy')

    Xs = [te_epoch0, te_epoch1, te_epoch2]#, te_epoch3, te_epoch4]
    Vs = [tr_epoch0, tr_epoch1, tr_epoch2]#, tr_epoch3, tr_epoch4]
    Ys = [te_loss0, te_loss1, te_loss2]#, te_loss3, te_loss4] 
    Zs = [tr_loss0, tr_loss1, tr_loss2]#, tr_loss3, tr_loss4] 
    colours_ = ['indianred', 'mediumpurple', 'skyblue']#, 'tomato', 'cyan']
    ls_ = ['-', '-', '-', '-', '-']
    labels_ =['Fixed SGD', 'Online SGD', 'Online SGLD']#, 'Online SGD per layer', 'Online SGLD per layer']
    fname = 'loss/cnn_loss_te_curve_comp.png' 
    lineplot(Xs, Ys, colours_, labels_, xlabel='Epoch', ylabel='Loss', fname=fname, ls=ls_, lw=3, logyscale=1)

    fname = 'loss/cnn_loss_tr_curve_comp.png' 
    lineplot(Vs, Zs, colours_, labels_, xlabel='Updates', ylabel='Loss', fname=fname, ls=ls_, lw=3, logyscale=1)


def diff_initial_lr(mlr, l2, num_epoch, batch_sz, batch_vl, model_type, opt_type, update_freq):

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

    colours = ['tomato', 'skyblue', 'green']

    for i,lr in enumerate([0.01, 0.001, 0.0001]):

        label = 'Init Lr. %f' % lr

        fdir = basepath +'/exp/mnist/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_0resetfreq/' % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq)
        lr_list = np.load(fdir+'lr.npy')
        l2_list = np.load(fdir+'l2.npy')
        dFlr_list = np.load(fdir+'dFdlr_list.npy')
        dFl2_list = np.load(fdir+'dFdl2_list.npy')
        tr_epoch = np.load(fdir+'tr_epoch.npy')
        te_epoch = np.load(fdir+'te_epoch.npy')
        tr_loss = np.load(fdir+'tr_loss.npy')
        te_loss = np.load(fdir+'te_loss.npy')

        ax00.plot(tr_epoch, tr_loss, color=colours[i], label=label, alpha=0.5, lw=5)
        ax01.plot(te_epoch, te_loss, color=colours[i], label=label, alpha=0.5, lw=5)
        ax10.plot(np.arange(len(lr_list)), lr_list, color=colours[i], label=label, alpha=0.5, lw=5)
        ax11.plot(np.arange(len(l2_list)), l2_list, color=colours[i], label=label, alpha=0.5, lw=5)

    ax00.legend()
    ax01.legend() 
    ax01.legend() 
    ax11.legend() 
    ax00.set_yscale('log')
    ax01.set_yscale('log')

    plt.suptitle('Initial Learning Rate Analysis')
    plt.tight_layout()
    plt.savefig('./figs/mnist/initial_lr_comparison_update_freq_%s_%s.png' % (model_type, opt_type), format='png')
    plt.close()


def opt_trloss_vs_vlloss(update_freq=1):

    opt_type = 'sgd'
    model_path = "/exp/mnist/mlr0.000000_lr0.010000_l20.000100/mlp_100epoch_100vlbz_%s_%dupdatefreq/" % (opt_type, 1)
    te_epoch3 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss3 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch3 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss3 = np.load(basepath + model_path+'tr_loss.npy')

    model_path = "/exp/mnist/mlr0.000000_lr0.001000_l20.000100/mlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq/" % (opt_type, 1)
    te_epoch0 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss0 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch0 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss0 = np.load(basepath + model_path+'tr_loss.npy')

    opt_type = 'sgd'
    model_path = "/exp/mnist/mlr0.000100_lr0.001000_l20.000100/mlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq/" % (opt_type, update_freq)
    te_epoch1 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss1 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch1 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss1 = np.load(basepath + model_path+'tr_loss.npy')

    opt_type = 'sgd'
    model_path = "/exp/mnist/mlr0.000100_lr0.001000_l20.000100/mlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq_trmetamobj/" % (opt_type, update_freq)
    te_epoch2 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss2 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch2 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss2 = np.load(basepath + model_path+'tr_loss.npy')

    Xs = [te_epoch3, te_epoch0, te_epoch1, te_epoch2]#, te_epoch3, te_epoch4]
    Vs = [tr_epoch3, tr_epoch0, tr_epoch1, tr_epoch2]#, tr_epoch3, tr_epoch4]
    Ys = [te_loss3 , te_loss0 , te_loss1, te_loss2]#, te_loss3, te_loss4] 
    Zs = [tr_loss3 , tr_loss0 , tr_loss1, tr_loss2]#, tr_loss3, tr_loss4] 
    colours_ = ['yellow', 'indianred', 'mediumpurple', 'darkmagenta']#, 'tomato', 'cyan']
    ls_ = ['-', '-', '-', '-', '-']
    labels_ =['Fixed SGD=0.01', 'Fixed SGD=0.001', 'Meta-Opt w/ Valid Grad', 'Meta-Opt w/ Train Grad']#, 'Online SGD per layer', 'Online SGLD per layer']
    fname = 'loss/mlp_trVSvlGrad_loss_te_curve_comp.pdf' 
    lineplot(Xs, Ys, colours_, labels_, xlabel='Epoch', ylabel='Loss', fname=fname, ls=ls_, lw=3, logyscale=1)

    fname = 'loss/mlp_trVSvlGrad_loss_tr_curve_comp.pdf' 
    print(fname)
    lineplot(Vs, Zs, colours_, labels_, xlabel='Updates', ylabel='Loss', fname=fname, ls=ls_, lw=3, logyscale=1)


def sensitiviy_init_learning_rate(update_freq=1):

    mlr = 0.000005
    #mlr = 0.00001
    mlr = 0.00005
    opt_type = 'sgd'
    Xs, Ys, Vs, Zs, Lrs, L2s, labels = [], [], [], [], [], [], []
    lr_list = [0.1,0.05,0.01,0.005,0.003, 0.001, 0.0007, 0.0005, 0.0001]
    for lr in lr_list:

        model_path = "/exp/mnist/mlr%f_lr%f_l20.000100/mlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq/" % (mlr, lr, opt_type, 1)
        te_epoch = np.load(basepath + model_path+'te_epoch.npy')
        te_loss = np.load(basepath + model_path+'te_loss.npy')
        tr_epoch = np.load(basepath + model_path+'tr_epoch.npy')
        tr_loss = np.load(basepath + model_path+'tr_loss.npy')
        dFlr_list = np.load(basepath + model_path+'dFdlr_list.npy')
        dFl2_list = np.load(basepath + model_path+'dFdl2_list.npy')

        Xs.append(te_epoch)
        Vs.append(tr_epoch)
        Ys.append(te_loss)
        Zs.append(tr_loss)
        Lrs.append(dFlr_list)
        L2s.append(dFl2_list)
        labels.append('Initi Lr=%f' % lr)

    Xsamples = np.arange(0,100,2)
    labels2 = ['Epoch %d' % x for x in Xsamples]
    Lrs = np.asarray(Lrs)
    L2s = np.asarray(L2s)
    dLr_wrt_lr = Lrs[:,Xsamples].T
    dL2_wrt_l2 = L2s[:,Xsamples].T

    colours_ = ['red', 'indianred', 'tomato', 'orange', 'brown', 'limegreen', 'green', \
                        'darkgreen', 'skyblue', 'cyan', 'navy', 'mediumpurple', 'black']
    ls_ = ['-'] * len(Xs)
    fname = 'mlp_initlr_loss_te_curve_comp_%s_mlr%f.pdf'  % (opt_type, mlr)
    lineplot(Xs, Ys, colours_, labels, xlabel='Epoch', ylabel='Loss', fname=fname, ls=ls_, lw=3, logyscale=1)

    fname = 'mlp_initlr_loss_tr_curve_comp_%s_mlr%f.pdf'  % (opt_type, mlr)
    lineplot(Vs, Zs, colours_, labels, xlabel='Updates', ylabel='Loss', fname=fname, ls=ls_, lw=3, logyscale=1)

    Us = [np.arange(len(vec)) for vec in Lrs]
    fname = 'mlp_initlr_dPdLr_curve_comp_%s_mlr%f.pdf'  % (opt_type, mlr)
    lineplot(Us, Lrs, colours_, labels, xlabel='Epoch', ylabel='|dPdLr|', fname=fname, ls=ls_, lw=3, logyscale=0, vmax=3500)

    fname = 'mlp_initlr_dPdLl2_curve_comp_%s_mlr%f.pdf'  % (opt_type, mlr)
    lineplot(Us, L2s, colours_, labels, xlabel='Epoch', ylabel='|dPdL2|', fname=fname, ls=ls_, lw=3, logyscale=0, vmax=257934)

    colors = [(1,0,0,alpha/100) for alpha in Xsamples]
    Gs = [lr_list for vec in Xsamples]
    ls_ = ['-'] * len(Xsamples)
    fname = 'mlp_dPdLr_wrt_lr_curve_comp_%s_mlr%f.pdf'  % (opt_type, mlr)
    lineplot(Gs, dLr_wrt_lr, colors, None, xlabel='Learning rate', ylabel='|dPdLr|', fname=fname, ls=ls_, lw=3, logyscale=0, logxscale=1, vmax=2500)

    fname = 'mlp_dPdLl2_wrt_l2_curve_comp_%s_mlr%f.pdf'  % (opt_type, mlr)
    lineplot(Gs, dL2_wrt_l2, colors, None, xlabel='Learning rate', ylabel='|dPdL2|', fname=fname, ls=ls_, lw=3, logyscale=0, logxscale=1, vmax=207934)

    
def sensitiviy_meta_lr(update_freq=1):

    lr = 0.0001
    opt_type = 'sgd'

    Xs, Ys, Vs, Zs, Lrs, L2s, labels = [], [], [], [], [], [], [] 
    mlr_list = [0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    for mlr in mlr_list:
        model_path = "/exp/mnist/mlr%f_lr%f_l20.000100/mlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq/" % (mlr, lr, opt_type, 1)
        te_epoch = np.load(basepath + model_path+'te_epoch.npy')
        te_loss = np.load(basepath + model_path+'te_loss.npy')
        tr_epoch = np.load(basepath + model_path+'tr_epoch.npy')
        tr_loss = np.load(basepath + model_path+'tr_loss.npy')
        dFlr_list = np.load(basepath + model_path+'dFdlr_list.npy')
        dFl2_list = np.load(basepath + model_path+'dFdl2_list.npy')

        Xs.append(te_epoch)
        Vs.append(tr_epoch)
        Ys.append(te_loss)
        Zs.append(tr_loss)
        Lrs.append(dFlr_list)
        L2s.append(dFl2_list)
        labels.append('Meta Lr=%f' % mlr)

    colours_ = ['tomato', 'green', 'skyblue', 'cyan', 'navy', 'mediumpurple', 'black']

    ls_ = ['-'] * len(Xs)
    fname = 'mlp_mlr_loss_te_curve_comp_%s_lr%f.pdf'  % (opt_type, lr)
    lineplot(Xs, Ys, colours_, labels, xlabel='Epoch', ylabel='Loss', fname=fname, ls=ls_, lw=3, logyscale=1)

    fname = 'mlp_mlr_loss_tr_curve_comp_%s_lr%f.pdf'  % (opt_type, lr)
    lineplot(Vs, Zs, colours_, labels, xlabel='Updates', ylabel='Loss', fname=fname, ls=ls_, lw=3, logyscale=1)

    Us = [np.arange(len(vec)) for vec in Lrs]
    fname = 'mlp_mlr_dPdLr_curve_comp_%s_lr%f.pdf'  % (opt_type, lr)
    lineplot(Us, Lrs, colours_, labels, xlabel='Epoch', ylabel='|dPdLr|', fname=fname, ls=ls_, lw=3, logyscale=0, vmax=2800)

    fname = 'mlp_mlr_dPdLl2_curve_comp_%s_lr%f.pdf'  % (opt_type, lr)
    lineplot(Us, L2s, colours_, labels, xlabel='Epoch', ylabel='|dPdL2|', fname=fname, ls=ls_, lw=3, logyscale=0, vmax=257934)

    Xsamples = np.arange(0,100,2)
    labels2 = ['Epoch %d' % x for x in Xsamples]
    Lrs = np.asarray(Lrs)
    L2s = np.asarray(L2s)
    dLr_wrt_lr = Lrs[:,Xsamples].T
    dL2_wrt_l2 = L2s[:,Xsamples].T

    colors = [(0,0,1,alpha/100) for alpha in Xsamples]
    Gs = [mlr_list for vec in Xsamples]
    ls_ = ['-'] * len(Xsamples)
    fname = 'mlp_dPdLr_wrt_mlr_curve_comp_%s_lr%f.pdf'  % (opt_type, lr)
    lineplot(Gs, dLr_wrt_lr, colors, None, xlabel='Meta learning rate', ylabel='|dPdLr|', fname=fname, ls=ls_, lw=3, logyscale=0, logxscale=1, vmax=2800)

    fname = 'mlp_dPdLl2_wrt_mlr_curve_comp_%s_lr%f.pdf'  % (opt_type, lr)
    lineplot(Gs, dL2_wrt_l2, colors, None, xlabel='Meta learning rate', ylabel='|dPdL2|', fname=fname, ls=ls_, lw=3, logyscale=0, logxscale=1, vmax=207934)


def get_all_model_performance():

    mlr = 0.0001
    opt_type = 'sgd'
    X1s, Y1s, V1s, Z1s, Lr1s, L21s, label1s, dPdLrs, dPdL2s, epochs, lrs, l2s = [], [], [], [], [], [], [], [], [], [], [], []
    lr_list = [0.1, 0.05,0.01,0.005,0.003, 0.001, 0.0007, 0.0005]
    l2_list = [0.0, 0.001, 0.0001, 0.00001]
    for lr in lr_list:
        for l2 in l2_list:
            model_path = "/exp/mnist/mlr%f_lr%f_l2%f/mlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                dFlr_list = np.load(basepath+model_path+'dFdlr_list.npy') 
                dFl2_list = np.load(basepath+model_path+'dFdl2_list.npy') 
                te_epoch = np.load(basepath+model_path+'te_epoch.npy')
                lr_list = np.load(basepath+model_path+'lr.npy')
                l2_list = np.load(basepath+model_path+'l2.npy')
                print(model_path, te_loss[-1])

                if not np.isnan(te_loss[-1]):
                    Y1s.append(te_loss[-1])
                    Z1s.append(tr_loss[-1])
                    dPdLrs.append(dFlr_list)
                    dPdL2s.append(dFl2_list)
                    epochs.append(te_epoch)
                    lrs.append(lr_list)
                    l2s.append(l2_list)
                    label1s.append('Init Lr=%f Init L2=%f' % (lr, l2))

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|dPdLr|')
    X = np.arange(len(dPdLrs[0]))
    for i, dPdLr_i in enumerate(dPdLrs):
        ax.plot(X, dPdLr_i/ 134794, color='salmon', ls='-')
    plt.tight_layout()
    plt.savefig('./figs/mnist/dPdLrs.pdf')

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Lr')
    X = np.arange(len(dPdLrs[0]))
    for i, lr_i in enumerate(lrs):
        ax.plot(X, lr_i, color='salmon', ls='-')
    plt.tight_layout()
    plt.savefig('./figs/mnist/mnist_Lrs.pdf')



    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|dPdL2|')
    for i, dPdL2_i in enumerate(dPdL2s):
        ax.plot(X, dPdL2_i/ 134794, color='deepskyblue', ls='-')
    plt.tight_layout()
    plt.savefig('./figs/mnist/dPdL2s.pdf')
    import pdb; pdb.set_trace()


    mlr = 0.0
    opt_type = 'sgd'
    X0s, Y0s, V0s, Z0s, Lr0s, L20s, label0s = [], [], [], [], [], [], []
    lr_list = [0.1, 0.05, 0.01, 0.001, 0.005, 0.0005]
    l2_list = [0.0, 0.001, 0.0001, 0.00001]
    for lr in lr_list:
        for l2 in l2_list:
            model_path = "/exp/mnist/mlr%f_lr%f_l2%f/mlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                if not np.isnan(te_loss[-1]):
                    Y0s.append(te_loss[-1])
                    Z0s.append(tr_loss[-1])
                    label0s.append('Init Lr=%f Init L2=%f' % (lr, l2))


    mlr = 0.0
    opt_type = 'sgd_step'
    X2s, Y2s, V2s, Z2s, Lr2s, L22s, label2s = [], [], [], [], [], [], []
    lr_list = [0.1, 0.05, 0.01, 0.001, 0.005, 0.0005]
    l2_list = [0.0, 0.001, 0.0001, 0.00001]
    for lr in lr_list:
        for l2 in l2_list:
            model_path = "/exp/mnist/mlr%f_lr%f_l2%f/mlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                if not np.isnan(te_loss[-1]):
                    Y2s.append(te_loss[-1])
                    Z2s.append(tr_loss[-1])
                    label2s.append('Init Lr=%f Init L2=%f' % (lr, l2))

    mlr = 0.0
    opt_type = 'sgd_expstep'
    X3s, Y3s, V3s, Z3s, Lr3s, L23s, label3s = [], [], [], [], [], [], []
    lr_list = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    l2_list = [0.0, 0.001, 0.0001, 0.00001]
    for lr in lr_list:
        for l2 in l2_list:
            model_path = "/exp/mnist/mlr%f_lr%f_l2%f/mlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                if not np.isnan(te_loss[-1]):
                    Y3s.append(te_loss[-1])
                    Z3s.append(tr_loss[-1])
                    label3s.append('Init Lr=%f Init L2=%f' % (lr, l2))

    mlr = 0.0
    opt_type = 'sgd_cosinestep'
    X5s, Y5s, V5s, Z5s, Lr5s, L25s, label5s = [], [], [], [], [], [], []
    lr_list = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    l2_list = [0.0, 0.001, 0.0001, 0.00001]
    for lr in lr_list:
        for l2 in l2_list:
            model_path = "/exp/mnist/mlr%f_lr%f_l2%f/mlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                if not np.isnan(te_loss[-1]):
                    Y5s.append(te_loss[-1])
                    Z5s.append(tr_loss[-1])
                    label5s.append('Init Lr=%f Init L2=%f' % (lr, l2))


    mlr = 0.00001
    opt_type = 'sgd'
    X4s, Y4s, V4s, Z4s, Lr4s, L24s, label4s = [], [], [], [], [], [], []
    lr_list = [0.1, 0.05,0.01,0.005,0.003, 0.001, 0.0007, 0.0005]
    l2_list = [0.0, 0.001, 0.0001, 0.00001]
    for lr in lr_list:
        for l2 in l2_list:
            model_path = "/exp/mnist/mlr%f_lr%f_l2%f/amlp_100epoch_100vlbz_%s_%dupdatefreq_0resetfreq/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                if not np.isnan(te_loss[-1]):
                    Y4s.append(te_loss[-1])
                    Z4s.append(tr_loss[-1])
                    label4s.append('Init Lr=%f Init L2=%f' % (lr, l2))


    ylabel = 'Test loss' 
    xlabel = 'Method'
    colours = ['tomato', 'deepskyblue', 'skyblue', 'mediumpurple', 'limegreen', 'green']
    Xticklabels = ['Fixed Lr\n L2 Coef.', 'Cosine Lr\n L2 Coef.', 'Step Lr\n L2 Coef.', 'Exp Lr\n L2 Coef.',  'Single\n Meta-opt', 'Full\n Meta-opt']
    fname = 'mnist/stability.pdf'
    Y_list = [Y0s, Y5s, Y2s, Y3s, Y1s, Y4s]
    violinplot(Xticklabels, Y_list, colours, xlabel, ylabel, fname)
   

def quotient_plot():

    ifold = 1
    labels = [1,2,3,4]
    avg_loss_list = []

    for quotient_i in labels:

        te_loss_list = []
        for ifold in [0,1,2,3,4,5,6,7,8,9]:
            #basepath = '/misc/vlgscratch4/ChoGroup/imj/'
            basepath = '/scratch/ji641/exp/'
            #qmlp_300epoch_100vlbz_sgd_1updatefreq_0resetfreq_fold0_quotient7
            modelpath = 'exp/mnist/mlr0.000001_lr0.100000_l20.000000/qmlp_100epoch_1000vlbz_sgd_1updatefreq_0resetfreq_fold%d_quotient%d/' 
            #modelpath = 'exp/mnist/mlr0.000010_lr0.100000_l20.000000/mlp_100epoch_100vlbz_sgd_1updatefreq_0resetfreq_1updatelabmda_fold%d_quotient%d/' 
            path = basepath + modelpath % (ifold, quotient_i)
            te_loss = np.load(path+'te_loss.npy')
            if ~np.isnan(te_loss[-1]) and te_loss[-1] < 0.5:
                te_loss_list.append(te_loss[-1]) 

        print(ifold, te_loss_list)
        avg_loss_list.append(te_loss_list)


    te_loss_list = []
    for ifold in [0,1,2,3,4,5,6,7,8,9]:
        basepath = '/scratch/ji641/imj/exp/' 
        modelpath = '/mnist/mlr0.000001_lr0.100000_l20.000000/amlp_100epoch_1000vlbz_sgd_1updatefreq_0resetfreq_fold%d/' 
        path = basepath + modelpath % (ifold)
        te_loss = np.load(path+'te_loss.npy')
        if ~np.isnan(te_loss[-1]) and te_loss[-1] < 0.5:
            te_loss_list.append(te_loss[-1]) 
    print(ifold, te_loss_list)
    avg_loss_list.append(te_loss_list)

    import pdb; pdb.set_trace()
    avg_loss_list = np.asarray(avg_loss_list).T
    #avg_loss_list = np.nanmean(avg_loss_list, axis=0)
    #, 10.006681
    time_list = np.asarray([0.282473, 0.282925, 0.287567, 0.289073, 0.292638, 0.35]) * 300 / 60
    time_list = np.asarray([0.282473, 0.282925, 0.287567, 0.289073, 0.35]) * 300 / 60
    fig, ax = plt.subplots(figsize=(16, 16))
    ax1 = ax.twinx()
    X = np.arange(len(avg_loss_list[0]))+1
    ax.boxplot(avg_loss_list)
    #ax.plot(X, avg_loss_list, 'o-', color='indianred', label='Test Loss')
    ax1.plot(X, time_list, 'x-', color='salmon', label='Training Time', lw=5, marker='.')
    ax1.set_ylabel('Time (h)')
    ax.set_ylabel('Test loss')
    ax.set_xlabel('Number of hyper-parameter sets')
    ax1.set_xticks(X)
    ax1.set_xticklabels((np.arange(len(avg_loss_list[0]))+1).tolist()[:-1] + ['Layerwise\n OHO'] )
    ax.legend()
    ax1.legend()
    fname = 'figs/mnist/hyperparam_time_tradeoff.pdf' 
    plt.tight_layout()
    plt.savefig(fname)
    print(fname)

    #time_list = np.asarray([0.561671, 6.445659, 6.750184, 6.767108, 6.773278, 6.787759, 6.809395, 10.122681]) * 300 / 60
    #X = np.arange(len(time_list))
    #fig, ax2 = plt.subplots(figsize=(16, 16))
    #ax2.plot(X, time_list, 'x-', color='salmon', label='Training Time', lw=5, marker='o')
    #ax2.set_ylabel('Time (h)')
    #ax2.set_xlabel('Number of Hyper-parameters')
    #Xlabel = [0,'Global OHO'] + ((np.arange(len(time_list)-2)+1)*3).tolist() + ['Full OHO']
    #print(Xlabel)
    #ax2.set_xticklabels(Xlabel)
    #fname = 'figs/cifar10/hyperparam_time_curve.pdf' 
    #plt.tight_layout()
    #plt.savefig(fname)





if __name__ == '__main__':

    #mo_shake_plot()
    #get_all_model_performance()
    #get_all_model_performance()

    opt_type = 'sgd'
    #model_path = "/exp/mnist/mlr0.000100_lr0.010000_l20.000100/amlp_100epoch_100vlbz_%s_1updatefreq/" % opt_type
    model_path = "/exp/mnist/mlr0.000100_lr0.001000_l20.000100/amlp_100epoch_100vlbz_%s_1updatefreq_0resetfreq/" % opt_type
    #main_layerwise_hyperparam(model_path, opt_type)

    opt_type = 'sgld'
    #model_path = "/exp/mnist/mlr0.000100_lr0.010000_l20.000100/amlp_100epoch_100vlbz_%s_1updatefreq_0resetfreq/" % opt_type
    model_path = "/exp/mnist/mlr0.000100_lr0.001000_l20.000100/amlp_100epoch_100vlbz_%s_1updatefreq_0resetfreq/" % opt_type
    #main_layerwise_hyperparam(model_path, opt_type)

    #sensitiviy_meta_lr(update_freq=1)
    #sensitiviy_init_learning_rate()
    #opt_trloss_vs_vlloss()
    quotient_plot()

    mlr = 0.00001
    lr = 0.001
    l2 = 0.000100
    num_epoch = 100 
    batch_sz = 100
    batch_vl = 100
    model_type = 'mlp'
    opt_type = 'sgd'
    update_freq = 1
    #main_single_lr(mlr, lr, l2, num_epoch, batch_sz, batch_vl, model_type, opt_type, update_freq)


    mlr = 0.00001
    lr = 0.0001
    l2 = 0.000100
    num_epoch = 100 
    batch_sz = 100
    batch_vl = 100
    model_type = 'mlp'
    opt_type = 'sgd'
    update_freq = 1
    #mnist_varying_jacob(mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq)
    #mnist_varying_update_freq(mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type)
    #diff_initial_lr(mlr, l2, num_epoch, batch_sz, batch_vl, model_type, opt_type, update_freq)

    lr = 0.1
    l2 = 0.000010
    model_type = 'mlp'
    opt_type = 'sgd'
    #gradient_correlation_analyis(lr, l2, model_type, num_epoch, batch_vl, opt_type)


    update_freq = 1
    #main_mlp_loss_comp(update_freq)

    mlr = 0.000004
    lr = 0.001
    l2 = 0.000010
    num_epoch = 100 
    batch_sz = 100
    batch_vl = 1000
    model_type = 'cnn'
    opt_type = 'sgd'
    update_freq = 1
    #main_single_lr(mlr, lr, l2, num_epoch, batch_sz, batch_vl, model_type, opt_type, update_freq)

    #main_cnn_loss_comp()
    pass





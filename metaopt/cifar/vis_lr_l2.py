import os, sys
from os import listdir
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from metaopt.visualize import *
from metaopt.util_stats import get_correlation
matplotlib.rcParams.update({'font.size': 32})
plt.rcParams.update({'font.size': 32})
basepath = '/misc/vlgscratch4/ChoGroup/imj/'
if 'CLUSTER' in os.environ: basepath = '/scratch/ji641/' 

def opt_trloss_vs_vlloss(update_freq=1):

    opt_type = 'sgd'
    model_path = "/exp/cifar10/mlr0.000000_lr0.100000_l20.000010/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq_1updatelabmda/" % (opt_type, 1)
    te_epoch0 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss0 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch0 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss0 = np.load(basepath + model_path+'tr_loss.npy')

    opt_type = 'sgd'
    model_path = "/exp/cifar10/mlr0.000010_lr0.100000_l20.000000/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq_1updatelabmda_fold0/" % (opt_type, update_freq)
    te_epoch1 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss1 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch1 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss1 = np.load(basepath + model_path+'tr_loss.npy')

    opt_type = 'sgd'
    model_path = "/exp/cifar10/mlr0.000010_lr0.100000_l20.000000/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq_1updatelabmda_trmetamobj_fold0/" % (opt_type, update_freq)
    te_epoch2 = np.load(basepath + model_path+'te_epoch.npy')
    te_loss2 = np.load(basepath + model_path+'te_loss.npy')
    tr_epoch2 = np.load(basepath + model_path+'tr_epoch.npy')
    tr_loss2 = np.load(basepath + model_path+'tr_loss.npy')

    Xs = [te_epoch0, te_epoch1, te_epoch2]#, te_epoch3, te_epoch4]
    Vs = [tr_epoch0, tr_epoch1, tr_epoch2]#, tr_epoch3, tr_epoch4]
    Ys = [te_loss0 , te_loss1, te_loss2]#, te_loss3, te_loss4] 
    Zs = [tr_loss0 , tr_loss1, tr_loss2]#, tr_loss3, tr_loss4] 
    colours_ = ['indianred', 'mediumpurple', 'darkmagenta']#, 'tomato', 'cyan']
    ls_ = ['-', '-', '-', '-', '-']
    labels_ =['Fixed SGD=0.1 L2=0.00001', 'Meta-Opt w/ Valid Grad', 'Meta-Opt w/ Train Grad']#, 'Online SGD per layer', 'Online SGLD per layer']
    fname = 'rez18_trVSvlGrad_loss_te_curve_comp.pdf' 
    lineplot(Xs, Ys, colours_, labels_, xlabel='Epoch', ylabel='Loss', fname=fname, ls=ls_, lw=5, logyscale=1)

    fname = 'rez18_trVSvlGrad_loss_tr_curve_comp.pdf' 
    print(fname)
    lineplot(Vs, Zs, colours_, labels_, xlabel='Updates', ylabel='Loss', fname=fname, ls=ls_, lw=5, logyscale=1)




def gradient_correlation_analyis(lr, l2, model_type, num_epoch, batch_vl, opt_type):

    basepath = '/scratch/ji641/'
    fig1, ax1 = plt.subplots(figsize=(16, 16))
    ax1.set_xlabel('Correlation Average ')
    ax1.set_ylabel('Correlation Standard Deviation')

    count = 0
    labels = ['SGD (fixed lr=0.1)', 'OMO']
    colours = ['tomato', 'indianred', 'skyblue', 'blue']
    tr_loss_list, te_loss_list, lr_list, l2_list = [], [], [], []
    for i, mlr in enumerate([0.0, 0.00001]):

        for dataset, num_epoch, model_type, batch_vl, l2 in [('MNIST', 100, 'mlp', 100, 0.000010), ('CIFAR10', 300, 'rez18', 1000, 0.0)]:            
            if dataset == 'MNIST':
                fdir = basepath +'/imj/exp/mnist/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq/' \
                            % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq, 0)
            else:
                fdir = basepath +'/exp/cifar10/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_1updatelabmda_fold0/' \
                            % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq, 0)
            label = dataset + ' ' + labels[i]
            tr_corr_mean_list= np.load(fdir+'tr_grad_corr_mean.npy')
            tr_corr_std_list = np.load(fdir+'tr_grad_corr_std.npy')
            ax1.plot(tr_corr_mean_list, tr_corr_std_list, color=colours[count], label=label, alpha=0.5, lw=3, marker='.')
            ax1.scatter(tr_corr_mean_list[0], tr_corr_std_list[0], color=colours[count], alpha=0.85, s=7, marker='*')
            ax1.annotate('Start', xy=(tr_corr_mean_list[0], tr_corr_std_list[0]), xytext=(0, 3),  # 3 points vertical offset
                                                    color=colours[count], textcoords="offset points",ha='center', va='bottom')
            count += 1
    ax1.legend() 
    #ax1.set_xscale('log')

    plt.tight_layout()
    plt.savefig('./figs/gradient_correlation_%s.pdf' % (opt_type), format='pdf')
    plt.close()




def mo_shake_plot(stype='shakeU'):

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
    colours = ['salmon', 'tomato', 'orangered', 'indianred', 'chocolate', 'firebrick']
    colours = ['salmon', 'indianred', 'chocolate', 'orange', 'yellowgreen']
    colours = ['salmon', 'limegreen', 'deepskyblue', 'mediumpurple', 'gray']

    #colours = ['tomato', 'orange', 'limegreen', 'skyblue', 'mediumpurple', 'gray']
    tr_loss_list, te_loss_list, lr_list, l2_list = [], [], [], []
    #for i, update_freq in enumerate([1, 10, 100]):
    for i, (shake, marker) in enumerate(zip([30, 50, 90, 180, 270], ['X','>','+','o','^'])):

        if stype == 'shakeU':
            fdir = basepath +'/imj/exp/cifar10/%s%d/mlr0.000000_lr0.100000_l20.000000/rez18_300epoch_1000vlbz_sgd_1updatefreq_0resetfreq_1updatelabmda/' % (stype, shake)
        else:
            fdir = basepath +'/imj/exp/cifar10/%s%d/mlr0.000000_lr0.100000_l20.000300/rez18_300epoch_1000vlbz_sgd_1updatefreq_0resetfreq_1updatelabmda/' % (stype, shake)
        tr_epoch = np.load(fdir+'tr_epoch.npy')
        te_epoch = np.load(fdir+'te_epoch.npy')

        tr_loss = np.load(fdir+'tr_loss.npy')
        te_loss = np.load(fdir+'te_loss.npy')
        lr_ = np.load(fdir+'lr.npy')
        l2_ = np.load(fdir+'l2.npy')

        if i == 0:
            ax00.plot(tr_epoch, tr_loss, color=colours[i], label='Halt OHO', alpha=0.75, lw=2, ls='--')
            ax01.plot(te_epoch, te_loss, color=colours[i], label='Halt OHO', alpha=0.75, lw=4, ls='--')
            #ax10.plot(np.arange(len(lr_)), lr_, color=colours[i], label='Epoch30 (SGD)', alpha=0.5, lw=3, ls='--')
            #ax11.plot(np.arange(len(l2_)), l2_, color=colours[i], label='Epoch30 (OHO)', alpha=0.5, lw=3, ls='--')
        else:
            ax00.plot(tr_epoch, tr_loss, color=colours[i], alpha=0.75, lw=2, ls='--')
            ax01.plot(te_epoch, te_loss, color=colours[i], alpha=0.75, lw=5, ls='--')
            #ax10.plot(np.arange(len(lr_)), lr_, color=colours[i], alpha=0.5, lw=3, ls='--')
            #ax11.plot(np.arange(len(l2_)), l2_, color=colours[i], alpha=0.5, lw=3, ls='--')

        ax00.axvline(x=shake, color='black', ls='--', alpha=0.35)
        ax01.axvline(x=shake, color='black', ls='--', alpha=0.35)
        ax10.axvline(x=shake, color='black', ls='--', alpha=0.35)
        ax11.axvline(x=shake, color='black', ls='--', alpha=0.35)

        fdir = basepath +'/imj/exp/cifar10/%s%d/mlr0.000010_lr0.100000_l20.000000/rez18_300epoch_1000vlbz_sgd_1updatefreq_0resetfreq_1updatelabmda/' % (stype, shake)
        #fdir = basepath +'/imj/exp/mnist/shakeD%d/mlr0.000010_lr0.001000_l20.000000/cnn_100epoch_100vlbz_sgd_1updatefreq_0resetfreq/' % shake
        tr_epoch = np.load(fdir+'tr_epoch.npy')
        te_epoch = np.load(fdir+'te_epoch.npy')

        tr_loss = np.load(fdir+'tr_loss.npy')
        te_loss = np.load(fdir+'te_loss.npy')
        lr_ = np.load(fdir+'lr.npy')
        l2_ = np.load(fdir+'l2.npy')

        if i == 0:
            ax00.plot(tr_epoch, tr_loss, color=colours[i], label='Cont. OHO', alpha=0.75, lw=1, marker=marker)
            ax01.plot(te_epoch, te_loss, color=colours[i], label='Cont. OHO', alpha=0.75, lw=6, marker=marker)
            ax10.plot(np.arange(len(lr_)), lr_, color=colours[i], label='Epoch30 ', alpha=0.75, lw=3, marker=marker)
            ax11.plot(np.arange(len(l2_)), l2_, color=colours[i], label='Epoch30 (OHO)', alpha=0.75, lw=3, marker=marker)
        else:
            ax00.plot(tr_epoch, tr_loss, color=colours[i], label=None, alpha=0.75, lw=1, marker=marker)
            ax01.plot(te_epoch, te_loss, color=colours[i], label=None, alpha=0.75, lw=6, marker=marker)
            ax10.plot(np.arange(len(lr_)), lr_, color=colours[i], label='Epoch%d' % shake, alpha=0.75, lw=4, marker=marker)
            ax11.plot(np.arange(len(l2_)), l2_, color=colours[i], label='Epoch%d' % shake, alpha=0.75, lw=4, marker=marker)

    ax00.set_xlim([0,300])
    ax01.set_xlim([0,300])
    ax00.set_ylim([0.2,0.7])
    ax01.set_ylim([0.2,0.7])

    ax00.legend()
    ax01.legend() 
    ax10.legend() 
    ax11.legend() 
    #ax00.set_yscale('log')
    #ax01.set_yscale('log')

    #plt.suptitle('Learning Rate Change')
    fig1.tight_layout()
    fig1.savefig('./figs/random_lrChnage_%s_tr.pdf' % stype, format='pdf')
    fig2.tight_layout()
    fig2.savefig('./figs/random_lrChnage_%s_te.pdf' % stype, format='pdf')
    fig3.tight_layout()
    fig3.savefig('./figs/random_lrChnage_%s_lr.pdf' % stype, format='pdf')
    fig3.tight_layout()
    fig4.savefig('./figs/random_lrChnage_%s_l2.pdf' % stype, format='pdf')
    fig4.tight_layout()

    plt.close()

def cifar_varying_resset_freq(mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq=1):

    tr_loss_list, te_loss_list, lr_list, l2_list = [], [], [], []
    colours = ['tomato', 'orange', 'limegreen', 'skyblue', 'mediumpurple', 'gray']
    colours = ['magenta', 'tomato', 'limegreen', 'skyblue', 'mediumpurple', 'gray']

    fig, ax00 = plt.subplots(figsize=(16, 16), sharey=False)
    ax00.set_xlabel('Updates')
    ax00.set_ylabel('Train Loss')
    for i, reset_freq in enumerate([0, 1, 10, 25, 50, 75]):
        fdir = basepath +'/exp/cifar10/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_%dupdatelabmda_fold0/' \
            % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq, reset_freq, update_lambda)
        label = 'No Reset' if i == 0 else 'Reset every %d' % reset_freq 
        tr_epoch = np.load(fdir+'tr_epoch.npy')
        tr_loss = np.load(fdir+'tr_loss.npy')
        ax00.plot(tr_epoch, tr_loss, color=colours[i], label=label, alpha=0.5, lw=5)
    ax00.legend()
    ax00.set_yscale('log')
    plt.tight_layout()
    plt.savefig('./figs/cifar10/learning_comparison_reset_freq_%s_%s_trloss.pdf' % (model_type, opt_type), format='pdf')
    plt.close()
    
    fig, ax01 = plt.subplots(figsize=(16, 16), sharey=False)
    ax01.set_xlabel('Epoch')
    ax01.set_ylabel('Test Loss')
    for i, reset_freq in enumerate([0, 1, 10, 25, 50, 75]):
        fdir = basepath +'/exp/cifar10/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_%dupdatelabmda_fold0/' \
            % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq, reset_freq, update_lambda)
        label = 'No Reset' if i == 0 else 'Reset every %d' % reset_freq 
        te_epoch = np.load(fdir+'te_epoch.npy')
        te_loss = np.load(fdir+'te_loss.npy')
        ax01.plot(te_epoch, te_loss, color=colours[i], label=label, alpha=0.5, lw=5)
    ax01.legend() 
    ax01.set_yscale('log')
    plt.tight_layout()
    plt.savefig('./figs/cifar10/learning_comparison_reset_freq_%s_%s_teloss.pdf' % (model_type, opt_type), format='pdf')
    plt.close()

    fig, ax10 = plt.subplots(figsize=(16, 16), sharey=False)
    ax10.set_xlabel('Epoch')
    ax10.set_ylabel('Learning Rates')
    for i, reset_freq in enumerate([0, 1, 10, 25, 50, 75]):
        fdir = basepath +'/exp/cifar10/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_%dupdatelabmda_fold0/' \
            % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq, reset_freq, update_lambda)
        label = 'No Reset' if i == 0 else 'Reset every %d' % reset_freq 
        lr_ = np.load(fdir+'lr.npy')
        ax10.plot(np.arange(len(lr_)), lr_, color=colours[i], label=label, alpha=0.5, lw=5)
    ax10.legend() 
    #plt.suptitle('Hyper-paramter Update Frequency Analysis')
    plt.tight_layout()
    plt.savefig('./figs/cifar10/learning_comparison_reset_freq_%s_%s_lr.pdf' % (model_type, opt_type), format='pdf')
    plt.close()

    fig, ax11 = plt.subplots(figsize=(16, 16), sharey=False)
    ax11.set_xlabel('Epoch')
    ax11.set_ylabel('L2 Weight Decay')
    for i, reset_freq in enumerate([0, 1, 10, 25, 50, 75]):
        fdir = basepath +'/exp/cifar10/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_%dupdatelabmda_fold0/' \
            % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq, reset_freq, update_lambda)
        if i == 0:
            label = 'No Reset'
        else:
            label = 'Reset every %d' % reset_freq 
        l2_ = np.load(fdir+'l2.npy')
        ax11.plot(np.arange(len(l2_)), l2_, color=colours[i], label=label, alpha=0.5, lw=5)
    ax11.legend() 
    #plt.suptitle('Hyper-paramter Update Frequency Analysis')
    plt.tight_layout()
    plt.savefig('./figs/cifar10/learning_comparison_reset_freq_%s_%s_l2.pdf' % (model_type, opt_type), format='pdf')
    plt.close()
    

def cifar_varying_update_freq(mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type):

    colours = ['tomato', 'skyblue', 'mediumpurple', 'green']

    fig, ax = plt.subplots(1, 1, figsize=(16, 16), sharey=False)
    ax.set_xlabel('Updates')
    ax.set_ylabel('Train Loss')

    te_loss_list_X = []
    X = [1, 10, 100, 1000]
    #for i, update_freq in enumerate([1, 10, 100]):
    for i, update_freq in enumerate(X):

        tr_loss_list, te_loss_list, lr_list, l2_list = [], [], [], []
        for ifold in range(9):
            update_lambda = 1 if update_freq == 0 else 0
            fdir = basepath +'/exp/cifar10/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_%dupdatelabmda_fold%d/' \
                % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq, 0, 1, ifold)
            label = 'Update every %d' % update_freq
            tr_epoch = np.load(fdir+'tr_epoch.npy')
            te_epoch = np.load(fdir+'te_epoch.npy')

            tr_loss = np.load(fdir+'tr_loss.npy')
            te_loss = np.load(fdir+'te_loss.npy')
            lr_ = np.load(fdir+'lr.npy')
            l2_ = np.load(fdir+'l2.npy')

            tr_loss_list.append(tr_loss)
            te_loss_list.append(te_loss[-1])
        te_loss_list_X.append(te_loss_list)
        #ax00.plot(tr_epoch, tr_loss, color=colours[i], label=label, alpha=0.5, lw=5)
    ax.boxplot(te_loss_list_X)
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(X)
    ax.legend()
    ax.legend()
    fname = 'figs/cifar/update_freq_box.pdf' 
    plt.tight_layout()
    plt.savefig(fname)
    print(fname)
    import pdb; pdb.set_trace()  



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
    #for i, update_freq in enumerate([1, 10, 100]):
    for i, update_freq in enumerate([1, 5, 15, 20]):

        update_lambda = 1 if update_freq == 0 else 0
        fdir = basepath +'/exp/cifar10/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_%dupdatelabmda_fold0/' \
            % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq, update_lambda, 0)
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
    plt.savefig('./figs/cifar10/learning_comparison_update_freq_%s_%s.png' % (model_type, opt_type), format='png')
    plt.close()
    


'''
def cifar_varying_update_freq(mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type):

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
    #for i, update_freq in enumerate([1, 10, 100]):
    for i, update_freq in enumerate([1, 5, 15, 20, 25, 30, 35, 40, 45, 50]):

        update_lambda = 1 if update_freq == 0 else 0
        fdir = basepath +'/exp/cifar10/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_%dupdatelabmda_fold0/' \
            % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq, 0, 1)
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
    plt.savefig('./figs/cifar10/learning_comparison_update_freq_%s_%s.png' % (model_type, opt_type), format='png')
    plt.close()
''' 


def main_loss_comp(model_path_str, hyper_list, label_str, fname_str, model='rez18', include_layerwiseF=0):

    Xs,Vs,Ys,Zs,Rs,Ss,Lrs,L2s,Us = [], [], [], [], [], [], [], [], []
    opt_type = 'sgd'
    for lr, l2 in [(0.15, 0.00001), (0.01, 0.00001)]:
        model_path = "/exp/cifar10/mlr0.000000_lr%f_l2%f/%s_300epoch_1000vlbz_%s_1updatefreq_0resetfreq_1updatelabmda/" % (lr,l2,model,opt_type)
        te_epoch0 = np.load(basepath + model_path+'te_epoch.npy')
        te_loss0 = np.load(basepath + model_path+'te_loss.npy')
        tr_epoch0 = np.load(basepath + model_path+'tr_epoch.npy')
        tr_loss0 = np.load(basepath + model_path+'tr_loss.npy')
        te_acc0 = np.load(basepath + model_path+'te_acc.npy')
        tr_acc0 = np.load(basepath + model_path+'tr_acc.npy')
        lr_ = np.load(basepath + model_path+'lr.npy')
        l2_ = np.load(basepath + model_path+'l2.npy')

        Xs.append(te_epoch0)
        Vs.append(tr_epoch0)
        Ys.append(te_loss0)
        Zs.append(tr_loss0)
        Ss.append(tr_acc0)
        Rs.append(te_acc0)
        Lrs.append(lr_)
        L2s.append(l2_)
        Us = [np.arange(len(lr_))]
    labels = ['SGD=0.15 L2=5e-4 (Fixed)', 'SGD=0.01 L2=1e-5 (Fixed)']

    for opt_type  in ['sgd_step', 'sgd_expstep']:
        model_path = "/exp/cifar10/mlr0.000000_lr0.100000_l20.000000/%s_300epoch_100vlbz_%s_1updatefreq_0resetfreq_1updatelabmda/" % (model,opt_type)
        te_epoch = np.load(basepath + model_path+'te_epoch.npy')
        te_loss = np.load(basepath + model_path+'te_loss.npy')
        tr_epoch = np.load(basepath + model_path+'tr_epoch.npy')
        tr_loss = np.load(basepath + model_path+'tr_loss.npy')
        te_acc = np.load(basepath + model_path+'te_acc.npy')
        tr_acc = np.load(basepath + model_path+'tr_acc.npy')
        lr_ = np.load(basepath + model_path+'lr.npy')
        l2_ = np.load(basepath + model_path+'l2.npy')

        Xs.append(te_epoch)
        Vs.append(tr_epoch)
        Ys.append(te_loss)
        Zs.append(tr_loss)
        Ss.append(tr_acc)
        Rs.append(te_acc)
        Lrs.append(lr_)
        L2s.append(l2_)
        Us.append(np.arange(len(lr_)))
        labels.append('%s=0.1' % (opt_type.split('_')[-1].upper()))

    models = [model, 'a'+model] if include_layerwiseF else [model] 

    opt_type = 'sgd'
    for model_name in models:
        for hyper_i in hyper_list:
            if type(hyper_i) == type(1):
                model_path = model_path_str % (model_name, hyper_i)
            else:
                model_path = model_path_str % (hyper_i, model_name)
            te_epoch = np.load(basepath + model_path+'te_epoch.npy')
            te_loss = np.load(basepath + model_path+'te_loss.npy')
            tr_epoch = np.load(basepath + model_path+'tr_epoch.npy')
            tr_loss = np.load(basepath + model_path+'tr_loss.npy')
            te_acc = np.load(basepath + model_path+'te_acc.npy')
            tr_acc = np.load(basepath + model_path+'tr_acc.npy')
            lr_ = np.load(basepath + model_path+'lr.npy')
            l2_ = np.load(basepath + model_path+'l2.npy')
           
            Xs.append(te_epoch)
            Vs.append(tr_epoch)
            Ys.append(te_loss)
            Zs.append(tr_loss)
            Ss.append(tr_acc)
            Rs.append(te_acc)
            if model_name.startswith('a'):
                labels.append('Layerwise-' + label_str % (hyper_i))
            else:
                Lrs.append(lr_)
                L2s.append(l2_)
                Us.append(np.arange(len(lr_)))
                #labels.append(label_str % (hyper_i))
                labels.append('Single-'+label_str % (hyper_i))
    #import pdb; pdb.set_trace()
    #colours_ = ['tomato', 'indianred', 'deepskyblue', 'skyblue', 'mediumpurple', \
    #                'darkviolet', 'purple', 'cyan', 'navy', 'mediumpurple', 'purple', 'black', 'blue', 'navy', ]
    colours_ = ['tomato', 'indianred', 'red', 'darksalmon', 'deepskyblue', 'skyblue', 'blue', 'mediumpurple', 'darkviolet',\
                         'purple', 'green', 'limegreen', 'lime', 'cyan', 'navy', 'black', 'blue', 'navy', ]

    ls_ = ['-'] * len(Xs)
    fname = fname_str  % (model, 'loss_te')
    print(fname)
    lineplot(Xs, Ys, colours_, labels, xlabel='Epoch', ylabel='Loss', fname=fname, ls=ls_, lw=5, logyscale=1)

    fname = fname_str  % (model, 'loss_tr')
    print(fname)
    lineplot(Vs, Zs, colours_, labels, xlabel='Updates', ylabel='Loss', fname=fname, ls=ls_, lw=5, logyscale=1)

    fname = fname_str  % (model, 'acc_te')
    print(fname)
    lineplot(Xs, Rs, colours_, labels, xlabel='Epoch', ylabel='Accuracy', fname=fname, ls=ls_, lw=5)

    fname = fname_str  % (model, 'acc_tr')
    print(fname)
    lineplot(Vs, Ss, colours_, labels, xlabel='Updates', ylabel='Accuracy', fname=fname, ls=ls_, lw=5)
    
    fname = fname_str  % (model, 'lr')
    print(fname)
    lineplot(Us, Lrs, colours_, labels, xlabel='Epoch', ylabel='Learning Rate', fname=fname, ls=ls_, lw=3)

    fname = fname_str % (model, 'l2')
    print(fname)
    lineplot(Us, L2s, colours_, labels, xlabel='Epoch', ylabel='L2 Coef.', fname=fname, ls=ls_, lw=3)



def main_single_lr(mlr, lr, l2, num_epoch, batch_sz, batch_vl, model_type, opt_type, update_freq):

    fdir = basepath +'/exp/cifar10/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_0resetfreq_1updatelabmda/' % (mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq)
    lr_list1 = np.load(fdir+'lr.npy')
    l2_list1 = np.load(fdir+'l2.npy')
    dFlr_list1 = np.load(fdir+'dFdlr_list.npy')
    dFl2_list1 = np.load(fdir+'dFdl2_list.npy')
    te_epoch1 = np.load(fdir+'te_epoch.npy')
    tr_data_list1 = np.load(fdir+'tr_loss.npy')
    te_data_list1 = np.load(fdir+'te_loss.npy')

    mlr = 0.0
    fdir = basepath +'/exp/cifar10/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_0resetfreq_1updatelabmda/' % (mlr, lr, 0.00001, model_type, num_epoch, batch_vl, 'sgd', update_freq)
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
    ax01.grid(0)

    color='indianred'
    ax01a = ax01.twinx()
    ax01a.set_ylabel('Learning Rate', color=color)
    ax01a.plot(epochs, lr_list1, color=color, lw=3)
    ax01a.grid(1)

    plt.tight_layout()
    plt.savefig('./figs/cifar10/learning_comparison_%s_%s.pdf' % (model_type, opt_type), format='pdf')
    plt.close()


def cifar10_mnist_mean_dLdsomething ():


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


    mlr = 0.00001
    opt_type = 'sgd'
    X2s, Y2s, V2s, Z2s, Lr2s, L22s, label2s, dPdLr2s, dPdL22s, epochs, lrs, l2s = [], [], [], [], [], [], [], [], [], [], [], []
    lr_list = [0.2, 0.15, 0.1, 0.05, 0.01, 0.005, 0.001]
    l2_list = [0.0, 0.0001, 0.00001]
    for lr in lr_list:
        for l2 in l2_list:
            model_path = "/exp/cifar10/mlr%f_lr%f_l2%f/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq_1updatelabmda/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                print(lr, l2, mlr, 1)
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')
                dFlr_list = np.load(basepath+model_path+'dFdlr_list.npy')
                dFl2_list = np.load(basepath+model_path+'dFdl2_list.npy')
                te_epoch = np.load(basepath+model_path+'te_epoch.npy')
                lr_i = np.load(basepath+model_path+'lr.npy')
                l2_i = np.load(basepath+model_path+'l2.npy')


                if not np.isnan(te_loss[-1]):
                    Y2s.append(te_loss[-1])
                    Z2s.append(tr_loss[-1])
                    dPdLr2s.append(dFlr_list)
                    dPdL22s.append(dFl2_list)
                    epochs.append(te_epoch)
                    lrs.append(lr_i)
                    l2s.append(l2_i)

                    label2s.append('Init Lr=%f Init L2=%f' % (lr, l2))

            model_path = "/exp/cifar10/mlr%f_lr%f_l2%f/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq_0updatelabmda/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                print(lr, l2, mlr, 2)
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')
                dFlr_list = np.load(basepath+model_path+'dFdlr_list.npy')
                dFl2_list = np.load(basepath+model_path+'dFdl2_list.npy')
                te_epoch = np.load(basepath+model_path+'te_epoch.npy')
                lr_i = np.load(basepath+model_path+'lr.npy')
                l2_i = np.load(basepath+model_path+'l2.npy')


                if not np.isnan(te_loss[-1]):
                    Y2s.append(te_loss[-1])
                    Z2s.append(tr_loss[-1])
                    dPdLr2s.append(dFlr_list)
                    dPdL22s.append(dFl2_list)
                    epochs.append(te_epoch)
                    lrs.append(lr_i)
                    l2s.append(l2_i)

                    label2s.append('Init Lr=%f Init L2=%f' % (lr, l2))
            else:
                print(model_path)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax1 = ax.twinx()

    ax.set_xlabel('Epoch')
    ax.set_ylabel('CIFAR10 |dPdLr|')
    ax1.set_ylabel('MNIST |dPdLr|')
    X = np.arange(len(dPdLr2s[0]))
    for i, dPdLr_i in enumerate(dPdLr2s):
        if i==0:
            ax.plot(X, dPdLr_i, color='indianred', ls='-', label='CIFAR10', lw=3) #/ 11173962
        else:
            ax.plot(X, dPdLr_i, color='indianred', ls='-', lw=3) #/ 11173962
    X = np.arange(len(dPdLrs[0]))
    for i, dPdLr_i in enumerate(dPdLrs):
        if i == 0:
            ax1.plot(X, dPdLr_i, color='salmon', ls='-', label='MNIST', lw=3) #/ 134794
        else:
            ax1.plot(X, dPdLr_i, color='salmon', ls='-', lw=3) #/ 134794

    ax1.legend()
    ax.legend()
    ax.set_yscale('log')
    ax1.set_yscale('log')

    plt.tight_layout()
    plt.savefig('./figs/dPdLrs_all.pdf')

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Lr')
    X = np.arange(len(dPdLr2s[0]))
    for i, lr_i in enumerate(lrs):
        ax.plot(X, lr_i, color='salmon', ls='-')
    plt.tight_layout()
    plt.savefig('./figs/cifar10/cifar10_Lrs.pdf')

    fig, ax = plt.subplots(figsize=(16, 16))
    ax1 = ax.twinx()

    ax.set_xlabel('Epoch')
    ax.set_ylabel('CIFAR10 |dPdL2|')
    ax1.set_ylabel('MNIST |dPdL2|')
    for i, dPdL2_i in enumerate(dPdL22s):
        if i ==0 :
            ax.plot(X, dPdL2_i, color='blue', ls='-', label='CIFAR10', lw=3) #/ 11173962
        else:
            ax.plot(X, dPdL2_i, color='blue', ls='-', lw=3) #/ 11173962
    X = np.arange(len(dPdLrs[0]))
    for i, dPdL2_i in enumerate(dPdL2s):
        if i ==0 :
            ax1.plot(X, dPdL2_i, color='deepskyblue', ls='-', label='MNIST', lw=3) #/ 134794
        else:
            ax1.plot(X, dPdL2_i, color='deepskyblue', ls='-', lw=3) #/ 134794
    ax1.legend()
    ax.legend()
    ax.set_yscale('log')
    ax1.set_yscale('log')
    plt.tight_layout()
    plt.savefig('./figs/dPdL2s_all.pdf')




def get_all_model_performance():

    mlr = 0.0001
    opt_type = 'sgd'
    X3s, Y3s, V3s, Z3s, Lr3s, L23s, label3s, dPdLrs, dPdL2s, epochs, lrs, l2s = [], [], [], [], [], [], [], [], [], [], [], []
    lr_list = [0.2, 0.15, 0.1, 0.05,0.01,0.005,0.001]
    l2_list = [0.0, 0.0001, 0.00001]
    for lr in lr_list:
        for l2 in l2_list:
            model_path = "/exp/cifar10/mlr%f_lr%f_l2%f/arez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq_1updatelabmda/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                dFlr_list = np.load(basepath+model_path+'dFdlr_list.npy') 
                dFl2_list = np.load(basepath+model_path+'dFdl2_list.npy') 
                te_epoch = np.load(basepath+model_path+'te_epoch.npy')
                lr_i = np.load(basepath+model_path+'lr.npy')
                l2_i = np.load(basepath+model_path+'l2.npy')

                if not np.isnan(te_loss[-1]):
                    Y3s.append(te_loss[-1])
                    Z3s.append(tr_loss[-1])
                    dPdLrs.append(dFlr_list)
                    dPdL2s.append(dFl2_list)
                    epochs.append(te_epoch)
                    lrs.append(lr_i)
                    l2s.append(l2_i)

                    label3s.append('Init Lr=%f Init L2=%f' % (lr, l2))

            #model_path = "/exp/cifar10/mlr%f_lr%f_l2%f/arez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq_0updatelabmda/" % (mlr, lr, l2, opt_type, 1)
            #if os.path.exists(basepath+model_path+'te_loss.npy'):
            #    te_loss = np.load(basepath + model_path+'te_loss.npy')
            #    tr_loss = np.load(basepath + model_path+'tr_loss.npy')

            #    if not np.isnan(te_loss[-1]):
            #        Y3s.append(te_loss[-1])
            #        Z3s.append(tr_loss[-1])
            #        label3s.append('Init Lr=%f Init L2=%f' % (lr, l2))

    mlr = 0.00001
    opt_type = 'sgd'
    X2s, Y2s, V2s, Z2s, Lr2s, L22s, label2s, dPdLr2s, dPdL22s, epochs, lrs, l2s = [], [], [], [], [], [], [], [], [], [], [], []
    lr_list = [0.2, 0.15, 0.1, 0.05, 0.01, 0.005, 0.001]
    l2_list = [0.0, 0.0001, 0.00001]
    for lr in lr_list:
        for l2 in l2_list:
            model_path = "/exp/cifar10/mlr%f_lr%f_l2%f/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq_1updatelabmda/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                print(lr, l2, mlr, 1)
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')
                dFlr_list = np.load(basepath+model_path+'dFdlr_list.npy')
                dFl2_list = np.load(basepath+model_path+'dFdl2_list.npy')
                te_epoch = np.load(basepath+model_path+'te_epoch.npy')
                lr_i = np.load(basepath+model_path+'lr.npy')
                l2_i = np.load(basepath+model_path+'l2.npy')


                if not np.isnan(te_loss[-1]):
                    Y2s.append(te_loss[-1])
                    Z2s.append(tr_loss[-1])
                    dPdLr2s.append(dFlr_list)
                    dPdL22s.append(dFl2_list)
                    epochs.append(te_epoch)
                    lrs.append(lr_i)
                    l2s.append(l2_i)

                    label2s.append('Init Lr=%f Init L2=%f' % (lr, l2))

            model_path = "/exp/cifar10/mlr%f_lr%f_l2%f/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq_0updatelabmda/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                print(lr, l2, mlr, 2)
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')
                dFlr_list = np.load(basepath+model_path+'dFdlr_list.npy')
                dFl2_list = np.load(basepath+model_path+'dFdl2_list.npy')
                te_epoch = np.load(basepath+model_path+'te_epoch.npy')
                lr_i = np.load(basepath+model_path+'lr.npy')
                l2_i = np.load(basepath+model_path+'l2.npy')


                if not np.isnan(te_loss[-1]):
                    Y2s.append(te_loss[-1])
                    Z2s.append(tr_loss[-1])
                    dPdLr2s.append(dFlr_list)
                    dPdL22s.append(dFl2_list)
                    epochs.append(te_epoch)
                    lrs.append(lr_i)
                    l2s.append(l2_i)

                    label2s.append('Init Lr=%f Init L2=%f' % (lr, l2))
            else:
                print(model_path)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|dPdLr|')
    X = np.arange(len(dPdLr2s[0]))
    for i, dPdLr_i in enumerate(dPdLr2s):
        ax.plot(X, dPdLr_i/ 11173962, color='salmon', ls='-')
    plt.tight_layout()
    plt.savefig('./figs/cifar10/dPdLrs.pdf')

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Lr')
    X = np.arange(len(dPdLr2s[0]))
    for i, lr_i in enumerate(lrs):
        ax.plot(X, lr_i, color='salmon', ls='-')
    plt.tight_layout()
    plt.savefig('./figs/cifar10/cifar10_Lrs.pdf')



    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|dPdL2|')
    for i, dPdL2_i in enumerate(dPdL22s):
        ax.plot(X, dPdL2_i/ 11173962, color='deepskyblue', ls='-')
    plt.tight_layout()
    plt.savefig('./figs/cifar10/dPdL2s.pdf')
    import pdb; pdb.set_trace()


    mlr = 0.0
    opt_type = 'sgd_step'
    X1s, Y1s, V1s, Z1s, Lr1s, L21s, label1s = [], [], [], [], [], [], []
    lr_list = [0.2, 0.15, 0.1, 0.05,0.01,0.005,0.001]
    l2_list = [0.0, 0.0001, 0.00001]
    for lr in lr_list:
        for l2 in l2_list:
            model_path = "/exp/cifar10/mlr%f_lr%f_l2%f/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq_1updatelabmda/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                if not np.isnan(te_loss[-1]):
                    Y1s.append(te_loss[-1])
                    Z1s.append(tr_loss[-1])
                    label1s.append('Init Lr=%f Init L2=%f' % (lr, l2))

            model_path = "/exp/cifar10/mlr%f_lr%f_l2%f/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                if not np.isnan(te_loss[-1]):
                    Y1s.append(te_loss[-1])
                    Z1s.append(tr_loss[-1])
                    label1s.append('Init Lr=%f Init L2=%f' % (lr, l2))

    mlr = 0.0
    opt_type = 'sgd_expstep'
    X4s, Y4s, V4s, Z4s, Lr4s, L24s, label4s = [], [], [], [], [], [], []
    lr_list = [0.2, 0.15, 0.1, 0.05,0.01,0.005,0.001]
    l2_list = [0.0, 0.0001, 0.00001]
    for lr in lr_list:
        for l2 in l2_list:
            model_path = "/exp/cifar10/mlr%f_lr%f_l2%f/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq_1updatelabmda/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                if not np.isnan(te_loss[-1]):
                    Y4s.append(te_loss[-1])
                    Z4s.append(tr_loss[-1])
                    label4s.append('Init Lr=%f Init L2=%f' % (lr, l2))

            model_path = "/exp/cifar10/mlr%f_lr%f_l2%f/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                if not np.isnan(te_loss[-1]):
                    Y4s.append(te_loss[-1])
                    Z4s.append(tr_loss[-1])
                    label4s.append('Init Lr=%f Init L2=%f' % (lr, l2))

    mlr = 0.0
    opt_type = 'sgd_cosinestep'
    X5s, Y5s, V5s, Z5s, Lr5s, L25s, label5s = [], [], [], [], [], [], []
    lr_list = [0.2, 0.15, 0.1, 0.05,0.01,0.005,0.001]
    l2_list = [0.0, 0.0001, 0.00001]
    for lr in lr_list:
        for l2 in l2_list:
            model_path = "/exp/cifar10/mlr%f_lr%f_l2%f/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq_1updatelabmda/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                if not np.isnan(te_loss[-1]):
                    Y5s.append(te_loss[-1])
                    Z5s.append(tr_loss[-1])
                    label5s.append('Init Lr=%f Init L2=%f' % (lr, l2))

            model_path = "/exp/cifar10/mlr%f_lr%f_l2%f/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                if not np.isnan(te_loss[-1]):
                    Y5s.append(te_loss[-1])
                    Z5s.append(tr_loss[-1])
                    label5s.append('Init Lr=%f Init L2=%f' % (lr, l2))


    mlr = 0.0
    opt_type = 'sgd'
    X0s, Y0s, V0s, Z0s, Lr0s, L20s, label0s = [], [], [], [], [], [], []
    lr_list = [0.2, 0.15, 0.1, 0.05,0.01,0.005,0.001]
    l2_list = [0.0, 0.001, 0.0001, 0.00001]
    for lr in lr_list:
        for l2 in l2_list:
            model_path = "/exp/cifar10/mlr%f_lr%f_l2%f/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq_1updatelabmda/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                if not np.isnan(te_loss[-1]):
                    Y0s.append(te_loss[-1])
                    Z0s.append(tr_loss[-1])
                    label0s.append('Init Lr=%f Init L2=%f' % (lr, l2))

            model_path = "/exp/cifar10/mlr%f_lr%f_l2%f/rez18_300epoch_1000vlbz_%s_%dupdatefreq_0resetfreq/" % (mlr, lr, l2, opt_type, 1)
            if os.path.exists(basepath+model_path+'te_loss.npy'):
                te_loss = np.load(basepath + model_path+'te_loss.npy')
                tr_loss = np.load(basepath + model_path+'tr_loss.npy')

                if not np.isnan(te_loss[-1]):
                    Y0s.append(te_loss[-1])
                    Z0s.append(tr_loss[-1])
                    label0s.append('Init Lr=%f Init L2=%f' % (lr, l2))


    ylabel = 'Test loss' 
    xlabel = 'Method'
    colours = ['tomato', 'indianred', 'skyblue', 'mediumpurple', 'limegreen']
    Xticklabels = ['Fixed Lr\n L2 Coef.', 'Cosine Lr\n L2 Coef.','Step Lr\n L2 Coef.', 'Exp Lr\n L2 Coef.',  'Single\n Meta-opt', 'Full\n meta-opt']
    fname = 'cifar10/stability.pdf'
    Y_list = [Y0s,Y5s, Y1s,  Y4s, Y2s, Y3s]
    violinplot(Xticklabels, Y_list, colours, xlabel, ylabel, fname)
    print(fname)


    ylabel = 'Test loss' 
    xlabel = 'Method'
    colours = ['tomato', 'indianred', 'skyblue', 'mediumpurple', 'limegreen']
    Xticklabels = ['Fixed Lr\n L2 Coef.', 'Cosine Lr\n L2 Coef.','Step Lr\n L2 Coef.', 'Exp Lr\n L2 Coef.',  'Single\n Meta-opt', 'Full\n meta-opt']
    fname = 'cifar10/stability_box.pdf'
    Y_list = [Y0s,Y5s, Y1s,  Y4s, Y2s, Y3s]
    violinplot(Xticklabels, Y_list, colours, xlabel, ylabel, fname)
    print(fname)


def layerwise_hyper_schdules(model_path, fname ):
   
    lr_list = np.load(basepath + model_path+'lr.npy').T
    l2_list = np.load(basepath + model_path+'l2.npy').T


    shapes = [(64, 3, 3, 3), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (128, 64, 3, 3), (128,), (128,), (128, 128, 3, 3), (128,), (128,), (128, 64, 1, 1), (128,), (128,), (128, 128, 3, 3), (128,), (128,), (128, 128, 3, 3), (128,), (128,), (256, 128, 3, 3), (256,), (256,), (256, 256, 3, 3), (256,), (256,), (256, 128, 1, 1), (256,), (256,), (256, 256, 3, 3), (256,), (256,), (256, 256, 3, 3), (256,), (256,), (512, 256, 3, 3), (512,), (512,), (512, 512, 3, 3), (512,), (512,), (512, 256, 1, 1), (512,), (512,), (512, 512, 3, 3), (512,), (512,), (512, 512, 3, 3), (512,), (512,), (10, 512), (10,)]


    colours=['red', 'red', 'indianred','indianred', 'magenta', 'magenta', 'tomato', 'tomato', 'orange', 'orange', \
                'goldenrod', 'goldenrod', 'yellow', 'yellow', 'yellowgreen', 'yellowgreen', 'limegreen', 'limegreen',\
                'green', 'green', 'darkgreen', 'darkgreen', 'skyblue','skyblue', 'cyan', 'cyan', \
                'deepskyblue', 'deepskyblue', 'dodgerblue', 'dodgerblue', 'blue','blue', 'navy', 'navy', \
                'purple', 'purple', 'mediumpurple', 'mediumpurple', 'indigo', 'indigo', \
                'slategray', 'slategray', 'darkslategray', 'darkslategray', 'gray', 'gray', 'black', 'black']
    count = 0
    labelWs, labelBs, l2Ws, ls = [], [], [], []
    lr_Ws, lr_Bs =[], []
    for lr, l2, shape in zip(lr_list, l2_list, shapes):

        if len(shape) > 1:
            subcount = 0
            count += 1
            labelWs.append('W%d' % count)
            ls.append('-')
            lr_Ws.append(lr)
            l2Ws.append(l2)
        else:
            subcount += 1
            labelBs.append('B%d%d' % (count, subcount))
            ls.append('--')
            lr_Bs.append(lr)
    
    #labels = ['W1', 'B1', 'W2', 'B2', 'W3', 'B3', 'W4', 'B4', 'W5', 'B5', 'W6', 'B6', 'W7', 'B7', 'W8', 'B8',\
    #           'W9', 'B9', 'W10', 'B10', 'W11', 'B11', 'W12', 'B12', 'W13', 'B13', 'W14', 'B14', 'W15', 'B15', \
    #           'W16', 'B16', 'W17', 'B17', 'W18', 'B18', 'W19', 'B19', 'W20', 'B20', 'W21', 'B21', 'W22', 'B22',\
    #           'W23', 'B23', 'W24', 'B24']
    #ls = ['-','--','-','--','-','--','-','--', '-','--','-','--','-','--','-','--', '-','--','-','--','-','--','-','--',\
    #        '-','--','-','--','-','--','-','--','-','--','-','--','-','--','-','--']

    N = len(colours)
    Xs = [np.arange(len(lr_list.T))]*N
    XWs = [np.arange(len(lr_list.T))]*len(lr_Ws)
    XBs = [np.arange(len(lr_list.T))]*len(lr_Bs)
    lr_Ws = np.asarray(lr_Ws)
    lr_Bs = np.asarray(lr_Bs)
    l2_Ws = np.asarray(l2Ws)

    plt.figure(figsize=(16, 16))
    corr_lr_l2 = get_correlation(lr_Ws.T, l2_Ws.T)
    plt.imshow(corr_lr_l2 , cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.xlabel('Learning Rate')
    plt.ylabel('L2 Coef.')
    plt.gca().invert_yaxis()
    plt.savefig("figs/cifar10/arez18/correlation_lr_l2_epoch.png")


    plt.figure(figsize=(16, 16))
    corr_lr_lr = get_correlation(lr_Ws.T, lr_Ws.T)
    plt.imshow(corr_lr_lr, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.xlabel('Learning Rate')
    plt.ylabel('Learning Rate')
    plt.gca().invert_yaxis()
    plt.savefig("figs/cifar10/arez18/correlation_lr_lr_epoch.png")

    plt.figure(figsize=(16, 16))
    corr_l2_l2 = get_correlation(l2_Ws.T, l2_Ws.T)
    plt.imshow(corr_l2_l2 , cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.xlabel('L2 Coef.')
    plt.ylabel('L2 Coef.')
    plt.gca().invert_yaxis()
    plt.savefig("figs/cifar10/arez18/correlation_l2_l2_epoch.png")

    plt.figure(figsize=(16, 16))
    corr_lrbias_lrbias = get_correlation(lr_Bs.T, lr_Bs.T)
    plt.imshow(corr_lrbias_lrbias, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.xlabel('Learning Rate for Bias')
    plt.ylabel('Learning Rate for Bias')
    plt.gca().invert_yaxis()
    plt.savefig("figs/cifar10/arez18/correlation_lrbias_lrbias_epoch.png")




    plt.figure(figsize=(16, 16))
    corr_lr_lr = get_correlation(lr_Ws, lr_Ws)
    plt.imshow(corr_lr_lr, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.xlabel('Learning Rate')
    plt.ylabel('Learning Rate')
    plt.gca().invert_yaxis()
    plt.savefig("figs/cifar10/arez18/correlation_lr_lr.png")

    plt.figure(figsize=(16, 16))
    corr_l2_l2 = get_correlation(l2_Ws, l2_Ws)
    plt.imshow(corr_l2_l2 , cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.xlabel('L2 Coef.')
    plt.ylabel('L2 Coef.')
    plt.gca().invert_yaxis()
    plt.savefig("figs/cifar10/arez18/correlation_l2_l2.png")

    plt.figure(figsize=(16, 16))
    corr_lrbias_lrbias = get_correlation(lr_Bs, lr_Bs)
    plt.imshow(corr_lrbias_lrbias, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.xlabel('Learning Rate for Bias')
    plt.ylabel('Learning Rate for Bias')
    plt.gca().invert_yaxis()
    plt.savefig("figs/cifar10/arez18/correlation_lrbias_lrbias.png")



    plt.figure(figsize=(16, 16))
    corr_lr_l2 = get_correlation(lr_Ws, l2_Ws)
    plt.imshow(corr_lr_l2 , cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.xlabel('Learning Rate')
    plt.ylabel('L2 Coef.')
    plt.gca().invert_yaxis()
    plt.savefig("figs/cifar10/arez18/correlation_lr_l2.png")

    plt.figure(figsize=(16, 16))
    corr_lr_lrbias = get_correlation(lr_Ws, lr_Bs)
    plt.imshow(corr_lr_lrbias, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.xlabel('Learning Rate')
    plt.ylabel('Learning Rate for Bias')
    plt.gca().invert_yaxis()
    plt.savefig("figs/cifar10/arez18/correlation_lr_lrbias.png")

    plt.figure(figsize=(16, 16))
    corr_lrbias_l2 = get_correlation(lr_Bs, l2_Ws)
    plt.imshow(corr_lrbias_l2, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.xlabel('Learning Rate for Bias')
    plt.ylabel('L2 Coef.')
    plt.gca().invert_yaxis()
    plt.savefig("figs/cifar10/arez18/correlation_lrbias_l2.png")


    fname_ = 'cifar10/arez18/lr_%s.png' % fname  
    xlabel = 'Epoch'
    ylabel = 'Learning Rate'
    lineplot(XWs, lr_Ws, colours[::2], labelWs, xlabel, ylabel, fname_, ls='-', lw=3, logyscale=0)
    #lineplot(Xs, lr_list[:N][::2], colours[::2], labels[:N][::2], xlabel, ylabel, fname_, ls='-', lw=3, logyscale=1)

    fname_ = 'cifar10/arez18/lrbias_%s.png' % fname  
    xlabel = 'Epoch'
    ylabel = 'Learning Rate for Bias'
    lineplot(XBs, lr_Bs, colours, None, xlabel, ylabel, fname_, ls='-', lw=3, logyscale=0)
    #lineplot(Xs, lr_list[:N][1::2], colours[1::2], labels[:N][1::2], xlabel, ylabel, fname_, ls='-', lw=3, logyscale=1)

    xlabel = 'Epoch'
    ylabel = 'L2 Coef.'
    fname_ = 'cifar10/arez18/l2_%s.png' % fname  
    lineplot(XWs, l2Ws, colours[::2], labelWs, xlabel, ylabel, fname_, ls='-', lw=3)


def random_search(ifold=4, sampler_type='uniform', num_trial=78):
  
    bookkeep = {'expstep':{'loss':[], 'epoch':[]}, \
                'cosine' :{'loss':[], 'epoch':[]}, \
                'step'   :{'loss':[], 'epoch':[]}, \
                'fixed'  :{'loss':[], 'epoch':[]},\
                'adam'   :{'loss':[], 'epoch':[]},\
                'metaopt':{'loss':[], 'epoch':[]}}
   
    time = {'fixed':0.31, 'metaopt':2.37}
    for trial in range(1,num_trial):
        path = '/scratch/ji641/cifar10/trial%d/%s/' % (trial, sampler_type)
        for fpath in listdir(path):
            mypath = path + fpath
            for fpath2 in listdir(mypath):
                mypath2 = mypath + '/'+fpath2
            
                #for ifold in [4,10]:
                ifold_str = 'fold%d' % ifold
                #if ((ifold_str in mypath2) or ('fold10' in mypath2)) \
                #                    and os.path.exists(mypath2+'/te_loss.npy'):
                if (ifold_str in mypath2) and os.path.exists(mypath2+'/te_loss.npy'):
                    
                    mlr_str = float(fpath.split('_')[0][3:])
                    epoch_str = fpath2.split('_')[1][:-5]
                    if 'expstep' in fpath2: #and ifold == 10:
                        key = 'expstep' 
                    elif 'cosine' in fpath2: # and ifold != 10:
                        key  = 'cosine'
                    elif 'sgd_step' in fpath2: # and ifold == 10:
                        key = 'step'
                    elif 'adam' in fpath2: # and ifold == 10:
                        key = 'adam'
                    elif mlr_str == 0.0: #and ifold != 10:
                        key = 'fixed'
                    elif mlr_str > 0 : # and ifold != 10:
                    #elif mlr_str > 0 and 'fold10' in mypath2: # and ifold != 10:
                        key = 'metaopt'
                        print(mypath2)
                    else:
                        key = None

                    if key is not None:
                        te_loss = np.load(mypath2 +'/te_loss.npy')
                        te_epoch = np.load(mypath2 +'/te_epoch.npy')
                        bookkeep[key]['loss'].append(te_loss[-1])
                        bookkeep[key]['epoch'].append(te_epoch[-1])
                    else:
                        print(mypath2)

    epoch_lists, num_trials, tot_times = [], [], []
    for key in bookkeep.keys():
        print(key)
        for i, val in enumerate(bookkeep[key]['loss']):
            if val < 0.30:
                break

        if key == 'metaopt':
            epoch_per_min = time[key]
            tot_time = epoch_per_min * np.sum(bookkeep[key]['epoch'][:i+1]) / 60
        else:
            epoch_per_min = time['fixed']
            tot_time = epoch_per_min * np.sum(bookkeep[key]['epoch'][:i+1]) / 60
            #tot_time = epoch_per_min * bookkeep[key]['epoch'][i] * (i+1) / 60
       
        print('%s: %d %f %fmin' % (key, i+1, val, tot_time))
        num_trials.append(i+1), 
        tot_times.append(tot_time)
        epoch_lists.append(bookkeep[key]['epoch'][i])

    labels = ['Fixed', 'Adam', 'Step', 'Coinse', 'Exp', 'Meta-Opt']
    num_trial_list = [num_trials[3], num_trials[4], num_trials[2], num_trials[1], num_trials[0], num_trials[5]]
    tot_time_list  = [tot_times[3], tot_times[4], tot_times[2], tot_times[1], tot_times[0], tot_times[5]]
    x = np.arange(len(bookkeep.keys()))
    fig, ax = plt.subplots(figsize=(16, 16))
    rects1 = ax.bar(x+.2, num_trial_list, width=0.5, color='tomato')
    ax.set_ylabel('# of Trials')
    #ax.set_title('Random search utill reaching 0.30 test loss')
    ax.set_ylim([0,150])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.savefig('figs/cifar10/randomsearch_%s_trial_performance.pdf' % sampler_type)

    fig, ax = plt.subplots(figsize=(16, 16))
    rects1 = ax.bar(x+.2, tot_time_list, width=0.5, color='skyblue')
    ax.set_ylabel('Wall clock Time (Hr)')
    #ax.set_title('Random search utill reaching 0.30 test loss')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0,150])
    fname = 'figs/cifar10/randomsearch_%s_time_performance.pdf' % sampler_type
    plt.savefig(fname)
    plt.tight_layout()
    print(fname)



def optimal_adaptive_params():

    te_loss_list = []
    basepath = '/misc/vlgscratch4/ChoGroup/imj/exp/cifar10/mlr0.000010_lr0.100000_l20.000000/'
    fdir = 'rez18_300epoch_1000vlbz_sgd_1updatefreq_0resetfreq_1updatelabmda/' 
    model_path = basepath + fdir
    te_loss = np.load(model_path+'te_loss.npy')
    te_loss_list.append(te_loss[-1]) 

    for d in [2,4,8,16]:
        fdir = 'qrez18_300epoch_1000vlbz_sgd_1updatefreq_0resetfreq_1updatelabmda_quotient%d/' % d
        model_path = basepath + fdir
        te_loss = np.load(model_path+'te_loss.npy')
        te_loss_list.append(te_loss[-1]) 

    basepath = '/misc/vlgscratch4/ChoGroup/imj/exp/cifar10/mlr0.000010_lr0.100000_l20.000000/'
    fdir = 'arez18_300epoch_1000vlbz_sgd_1updatefreq_0resetfreq_1updatelabmda/' 
    model_path = basepath + fdir
    te_loss = np.load(model_path+'te_loss.npy')
    te_loss_list.append(te_loss[-1]) 
    import pdb; pdb.set_trace()


def quotient_plot():

    ifold = 1
    labels = [1,2,4,6]
    avg_loss_list = []

    for quotient_i in labels:

        te_loss_list = []
        for ifold in [0,1,2,3,4,5,6,7,8,9]:
            #basepath = '/misc/vlgscratch4/ChoGroup/imj/'
            modelpath = 'exp/cifar10/mlr0.000005_lr0.010000_l20.000000/qrez18_300epoch_1000vlbz_sgd_1updatefreq_0resetfreq_1updatelabmda_fold%d_quotient%d/' 
            path = basepath + modelpath % (ifold, quotient_i)
            te_loss = np.load(path+'te_loss.npy')
            if ~np.isnan(te_loss[-1]) and te_loss[-1] < 1.5:
                te_loss_list.append(te_loss[-1]) 

        print(ifold, te_loss_list)
        avg_loss_list.append(te_loss_list)
    #avg_loss_list = np.asarray(avg_loss_list).T
    #avg_loss_list = np.nanmean(avg_loss_list, axis=0)

    te_loss_list = []
    for ifold in [0,1,2,3,4,5,6,7,8,9]:
        modelpath = '/exp/cifar10/mlr0.000010_lr0.100000_l20.000000/arez18_300epoch_1000vlbz_sgd_1updatefreq_0resetfreq_1updatelabmda_fold%d/'
        path = basepath + modelpath % (ifold)
        te_loss = np.load(path+'te_loss.npy')
        if ~np.isnan(te_loss[-1]) and te_loss[-1] < 1.5:
            te_loss_list.append(te_loss[-1]) 

        print(ifold, te_loss_list)
    avg_loss_list.append(te_loss_list)
    avg_loss_list = np.asarray(avg_loss_list).T
    labels.append('Layerwise\n OHO')

    #, 10.006681
    #time_list = np.asarray([np.nan, 6.699637, 6.770426, 6.771454, 6.773278, 6.787759, 6.809395]) * 300 / 60
    #time_list = np.asarray([np.nan, 6.445659, 6.750184, 6.767108, 6.773278, 6.787759, 6.809395]) * 300 / 60
    time_list = np.asarray([4.802622, 4.811060, 4.829998, 4.843670,6.378645]) * 300 / 60

    fig, ax = plt.subplots(figsize=(16, 16))
    ax1 = ax.twinx()
    X = np.arange(len(avg_loss_list))+1
    ax.boxplot(avg_loss_list)
    #ax.plot(X, avg_loss_list, 'o-', color='indianred', label='Test Loss')
    ax1.plot(X, time_list, 'x-', color='salmon', label='Training Time', lw=5, marker='.')
    ax1.set_ylabel('Time (h)')
    ax.set_ylabel('Test loss')
    ax.set_xlabel('Number of learning rates')
    #ax.set_title('Number of learning rates')
    ax.set_xticks(X)
    ax.set_xticklabels(labels)
    ax.legend()
    ax1.legend()
    fname = 'figs/cifar/hyperparam_time_tradeoff2.pdf' 
    plt.tight_layout()
    plt.savefig(fname)

    '''
    #time_list = np.asarray([0.561671, 6.445659, 6.750184, 6.767108, 6.773278, 6.787759, 6.809395, 10.122681]) * 300 / 60
    time_list = np.asarray([0.561671, 6.445659, 6.750184, 6.767108, 6.773278, 6.787759, 6.809395, 10.122681]) * 300 / 60
    X = np.arange(len(time_list))
    fig, ax2 = plt.subplots(figsize=(16, 16))
    ax2.plot(X, time_list, 'x-', color='salmon', label='Training Time', lw=5, marker='o')
    ax2.set_ylabel('Time (h)')
    ax2.set_xlabel('Number of Hyper-parameters')
    Xlabel = [0,'Global OHO'] + ((np.arange(len(time_list)-2)+1)*3).tolist() + ['Full OHO']
    print(Xlabel)
    ax2.set_xticklabels(Xlabel)
    fname = 'figs/cifar10/hyperparam_time_curve.pdf' 
    plt.tight_layout()
    plt.savefig(fname)
    '''


if __name__ == '__main__':

    #opt_trloss_vs_vlloss(update_freq=1)
    #mo_shake_plot('shakeU')
    #mo_shake_plot('shakeL2')

    model='rez18'
    opt_type = 'sgd'
    hyper_list = [1, 10, 50, 100, 1000]
    label_str = 'Update Freq %d'
    fname_str = '%s_updatefreq_%s_curve_comp.pdf' 
    model_path_str = "/exp/cifar10/mlr0.000010_lr0.100000_l20.000000/%s_300epoch_1000vlbz_"\
                                                                +opt_type+"_%dupdatefreq_0resetfreq/"
    #main_loss_comp(model_path_str, hyper_list, label_str, fname_str=fname_str, model=model)



    hyper_list = [100, 500, 1000]
    label_str = 'Validation batch size %d'
    fname_str = '%s_vlbatchsz_%s_curve_comp.pdf' 
    model_path_str = "/exp/cifar10/mlr0.000010_lr0.100000_l20.000000/%s_300epoch_%dvlbz_"\
                                                                +opt_type+"_1updatefreq_0resetfreq/"
    #main_loss_comp(model_path_str, hyper_list, label_str, fname_str=fname_str, model=model)


    #hyper_list = [0.0, 0.0001]
    #label_str = 'Initial l2 coef. %f'
    #fname_str = '%s_initl2_%s_curve_comp.png' 
    #model_path_str = "/exp/cifar10/mlr0.000010_lr0.100000_l2%f/"+model+"_300epoch_1000vlbz_"\
    #                                                            +opt_type+"_1updatefreq_0resetfreq/"
    #main_loss_comp(model_path_str, hyper_list, label_str, fname_str=fname_str, model=model)


    hyper_list = [0.1, 0.15, 0.2]
    label_str = 'Init Lr=%.2f  Init l2=0'
    fname_str = '%s_lr_%s_curve_comp.pdf' 
    model_path_str = "/exp/cifar10/mlr0.000010_lr%f_l20.000000/%s_300epoch_1000vlbz_"\
                                                                +opt_type+"_1updatefreq_0resetfreq_1updatelabmda/"
    #main_loss_comp(model_path_str, hyper_list, label_str, fname_str=fname_str, model=model, include_layerwiseF=1)


    hyper_list = [0.1, 0.15, 0.2]
    label_str = 'Init Lr=%.2f'
    fname_str = '%s_lr_nol2_%s_curve_comp.pdf' 
    model_path_str = "/exp/cifar10/mlr0.000010_lr%f_l20.000000/%s_300epoch_1000vlbz_"\
                                            +opt_type+"_1updatefreq_0resetfreq_0updatelabmda/"
    #main_loss_comp(model_path_str, hyper_list, label_str, fname_str=fname_str, model=model, include_layerwiseF=1)

    ###
    lr = 0.15
    model='arez18'
    update_lambda = 0
    fname = '%flr_0lambda_%dupdate_lambda' % (lr, update_lambda)
    model_path_str = "/exp/cifar10/mlr0.000010_lr%f_l20.000000/%s_300epoch_1000vlbz_"\
                                            +opt_type+"_1updatefreq_0resetfreq_%dupdatelabmda/"
    model_path = model_path_str % (lr, model, update_lambda)
    #layerwise_hyper_schdules(model_path, fname)

    lr = 0.1
    update_lambda = 1
    fname = '%flr_0lambda_%dupdate_lambda_500eopch' % (lr, update_lambda)
    model_path_str = "/exp/cifar10/mlr0.000010_lr%f_l20.000000/%s_500epoch_1000vlbz_"\
                                            +opt_type+"_1updatefreq_0resetfreq_%dupdatelabmda/"
    model_path = model_path_str % (lr, model, update_lambda)
    #layerwise_hyper_schdules(model_path, fname)

    update_lambda = 1
    fname = '%flr_0lambda_%dupdate_lambda_500eopch' % (lr, update_lambda)
    model_path = '/exp/cifar10/mlr0.000010_lr0.100000_l20.000000/arez18_500epoch_1000vlbz_sgd_1updatefreq_0resetfreq_1updatelabmda/'
    #layerwise_hyper_schdules(model_path, fname)

    update_lambda = 1
    fname = '%flr_0lambda_%dupdate_lambda_510eopch_fixedLr_l2dynamic' % (lr, update_lambda)
    model_path = '/exp/cifar10/mlr0.000010_lr0.100000_l20.000000/arez18_510epoch_1000vlbz_sgd_1updatefreq_0resetfreq_1updatelabmda/'
    ##layerwise_hyper_schdules(model_path, fname)

    update_lambda = 1
    fname = '%flr_0lambda_%dupdate_lambda_520eopch_Lrdynamic_l2fixed' % (lr, update_lambda)
    model_path = '/exp/cifar10/mlr0.000010_lr0.100000_l20.000000/arez18_520epoch_1000vlbz_sgd_1updatefreq_0resetfreq_1updatelabmda/'
    ##layerwise_hyper_schdules(model_path, fname)

    #cifar10_mnist_mean_dLdsomething ()
    #get_all_model_performance()
    #random_search(ifold=10, sampler_type='uniform', num_trial=78)
    #random_search(ifold=0, sampler_type='skopt', num_trial=43)
    #optimal_adaptive_params()
    #quotient_plot()

    mlr = 0.00001
    lr = 0.1
    l2 = 0.0#0.0001
    num_epoch=300
    batch_sz=100
    batch_vl=1000
    model_type='rez18'
    opt_type='sgd'
    update_freq=1
    #main_single_lr(mlr, lr, l2, num_epoch, batch_sz, batch_vl, model_type, opt_type, update_freq)
    #cifar_varying_update_freq(mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type)
    #cifar_varying_resset_freq(mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type, update_freq=1)

    model_type='arez18'
    cifar_varying_update_freq(mlr, lr, l2, model_type, num_epoch, batch_vl, opt_type)

    mlr = 0.00001
    lr = 0.1
    l2 = 0.0#0.0001
    num_epoch=300
    batch_sz=100
    batch_vl=1000
    model_type='rez18'
    opt_type='sgd'
    #gradient_correlation_analyis(lr, l2, model_type, num_epoch, batch_vl, opt_type)



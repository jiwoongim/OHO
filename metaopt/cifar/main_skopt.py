import os, sys, math, argparse, time
import torch
import torch.optim as optim
from skopt import gp_minimize
from main import main, parse_args
ifold=0

args = parse_args()
args.trial=0

def f(x, args=args):
    lr, l2, num_epoch = x
    args.lr = lr 
    args.lambda_l2 = l2
    args.num_epoch = num_epoch
    args.sampler_type = 'skopt'
    args.trial += 1
    args.is_cuda = 1
    args.save=1
    print('Sampler %s' % args.sampler_type)

    return main(args, trial=args.trial, ifold=ifold, device=device)


def f_step(x, args=args):
    lr, l2, num_epoch, step = x
    args.lr = lr 
    args.step_size = step
    args.lambda_l2 = l2
    args.num_epoch = num_epoch
    args.sampler_type = 'skopt'
    args.trial += 1
    args.is_cuda = 1
    args.save=1
    print('Sampler %s' % args.sampler_type)

    return main(args, trial=args.trial, ifold=ifold, device=device)


def fstep(x, args=args):
    lr, l2, num_epoch, step = x
    args.lr = lr 
    args.step_size = step
    args.lambda_l2 = l2
    args.num_epoch = num_epoch
    args.sampler_type = 'skopt'
    args.trial += 1
    args.is_cuda = 1
    args.save=1
    print('Sampler %s' % args.sampler_type)

    return main(args, trial=args.trial, ifold=ifold, device=device)

def f_expstep(x, args=args):
    lr, l2, num_epoch, gamma = x
    args.lr = lr 
    args.gamma = gamma
    args.lambda_l2 = l2
    args.num_epoch = num_epoch
    args.sampler_type = 'skopt'
    args.trial += 1
    args.is_cuda = 1
    args.save=1
    print('Sampler %s' % args.sampler_type)

    return main(args, trial=args.trial, ifold=ifold, device=device)

def f_adam(x, args=args):
    lr, l2, num_epoch, beta1, beta2 = x
    args.lr = lr 
    args.beta1 = beta1 
    args.beta2 = beta2
    args.lambda_l2 = l2
    args.num_epoch = num_epoch
    args.sampler_type = 'skopt'
    args.trial += 1
    args.is_cuda = 1
    args.save=1
    print('Sampler %s' % args.sampler_type)

    return main(args, trial=args.trial, ifold=ifold, device=device)




if __name__ == "__main__":
   
    is_cuda = 1 
    device = 'cuda' if is_cuda else 'cpu'
    print(args.opt_type)
    if args.opt_type == 'sgd_step':
        res = gp_minimize(f_step,                  # the function to minimize
                  [(0.0001, 0.2), (0,0.0002), (100,300), (100,5000)],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=50,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=ifold)   # the random seed

    elif args.opt_type == 'sgd_expstep':
        res = gp_minimize(f_expstep,                  # the function to minimize
                  [(0.0001, 0.2), (0,0.0002), (100,300), (0.1,0.9)],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=50,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=ifold)   # the random seed

    elif args.opt_type == 'adam':
        res = gp_minimize(f_adam,                  # the function to minimize
                  [(0.0001, 0.2), (0,0.0002), (100,300), (0.5,0.99), (0.5,0.99)],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=50,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=ifold)   # the random seed


    else:
        res = gp_minimize(f,                  # the function to minimize
                  [(0.0001, 0.2), (0,0.0002), (100,300)],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=50,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=ifold)   # the random seed





# Pytorch implementation of Thomspon Samling for linear bandits
# Ref: https://arxiv.org/pdf/1209.3352v4.pdf and https://proceedings.mlr.press/v28/agrawal13.pdf
# code based on: https://github.com/khashayarkhv/contextual-bandits/blob/master/scripts/runpriorfreeTS.m

import torch
import torch.distributions as dist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from concurrent.futures import ThreadPoolExecutor,as_completed,wait,ALL_COMPLETED
import threading
import argparse
import time
import math


def LinTS(T, k, d, R, delta, sigma_e, xmax, arm_para_noise, case):
    '''
    run LinTS algo
    T: time horizon
    k: number of arm
    d: number of covairiates with intercept
    R: a constant for R-sub-Gaussian
    delta: a constant
    sigma_e: subgaussianity parameter
    xmax: Maximum of l2-norm
    arm_para_noise: use true (1) noise or not, default false (0)
    case: 1: mixture, 2: N(1d, 1d), 3: N(0d, 1d)
    '''
    
    if arm_para_noise == 1:
        arm_para = np.random.normal(0,1, size = (k, d)) # k*d matrix
    else:
        if case == 1: # same as stated in paper, mixture
            indicator = np.random.random()
            if indicator > 0.5:
                arm_para = np.random.multivariate_normal(-np.ones(d), np.eye(d), k)
            else:
                arm_para = np.random.multivariate_normal(np.ones(d), np.eye(d), k)
        elif case == 2:
            arm_para = np.random.multivariate_normal(np.ones(d), np.eye(d), k) # case 2: ~N(1d, 1d)
        elif case == 3:
            arm_para = np.random.multivariate_normal(np.zeros(d), np.eye(d), k) # case 3: ~N(0d, 1d)
    
    arm_para = torch.from_numpy(arm_para).type(torch.float64) # k*d matrix
    reward_vec = torch.zeros(T, k, dtype = torch.float64)
    regret_vec = [0.] * T #torch.zeros((T,), dtype = torch.float64)
    B = torch.zeros(d, d, k, dtype = torch.float64)
    for i in range(k):
        B[:,:,i] = torch.eye(d)
    
    mu_hat = torch.zeros(d, k, dtype = torch.float64)
    f = torch.zeros(d, k, dtype = torch.float64)
    mu_tilda = torch.zeros(d, k, dtype = torch.float64)
    # residual = torch.zeros((T,), dtype = torch.float64)
    
    v = R*math.sqrt(9*k*d*math.log(T/delta))
    for t in range(T):
        X_ = dist.multivariate_normal.MultivariateNormal(torch.zeros(d), torch.eye(d)).sample().type(torch.float64) # size: d,
        X_ = X_.unsqueeze(0) # size: 1,d
        X = np.clip(X_, -xmax, xmax)
        X[0,0] = 1 # intercept
        x = X.transpose(0,1) # size: d,1
        
        for i in range(k):
            sampled = dist.multivariate_normal.MultivariateNormal(mu_hat[:,i], v**2*torch.inverse(B[:,:,i])).sample().type(torch.float64) # size: d,
            mu_tilda[:,i] = sampled
        
        round_res = x.transpose(0,1) @ mu_tilda # size: 1,k
        opt_arm = torch.argmax(round_res) # optiaml arm index
        
        gt_res = arm_para @ x # size: k,1
        #print(gt_res.shape)
        theo_reward, _ = torch.max(gt_res, dim = 0)
        round_reward = gt_res[opt_arm]
        
        # calculate regret
        if t < 1:
            regret_vec[t] = (theo_reward - round_reward).item()
        else:
            regret_vec[t] = regret_vec[t-1] + (theo_reward - round_reward).item()
        
        # reward
        reward_vec[t, opt_arm] = round_reward + sigma_e * dist.normal.Normal(0,1).sample().type(torch.float64)
        #print(reward_vec[t, opt_arm].shape)
        
        # now, update b, mu, f
        B[:,:,opt_arm] = B[:,:,opt_arm] + x @ x.transpose(0,1)
        
        #print(f[:,opt_arm].shape)
        f[:,opt_arm] = f[:,opt_arm] + reward_vec[t, opt_arm] * x.squeeze(1)
        
        mu_hat[:,opt_arm] = torch.inverse(B[:,:,opt_arm]) @ f[:,opt_arm]
        
    
    return regret_vec


def runOnce(total_regret_vec, T, k, d, R, delta, sigma_e, xmax, arm_para_noise, case, i):
    lock = threading.Lock()
    #total_regret_vec = np.zeros((1000, T))
    temp_reg = LinTS(T, k, d, R, delta, sigma_e, xmax, arm_para_noise, case)
    # when write data in the array, a thread lock is required
    lock.acquire()
    total_regret_vec[i,:] = temp_reg
    lock.release()


def simulate(T, d, R, delta, sigma_e, xmax, arm_para_noise, case, indi):
    np.random.seed(41)
    torch.manual_seed(41)
    # T = 100000
    # kappa1 = 1.1
    # d = 3 + 1
    # lmd = 0.1
    # delta = 1/T
    # m2 = 1
    # L = 1
    # sigma_e = 0.5
    # xmax = 1
    # arm_para_noise = 0
    # switch = 0
    # case = 3
    print(T, d, R, delta, sigma_e, xmax, arm_para_noise, case)
    print("over wirte k only")
    num2 = 0
    num_simu = 50
    #total_regret_vec = np.zeros((num_simu, T))
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=10))
    for k in [2, 5, 8, 10, 15]:
        total_regret_vec = np.zeros((num_simu, T))
        # enable multi-thread
        with ThreadPoolExecutor(max_workers=ncpus) as executor:
            task_all = [executor.submit(runOnce, total_regret_vec, T, k, d, R, delta, sigma_e, xmax, arm_para_noise, case, i) for i in range(num_simu)]
            for future in as_completed(task_all):
                data = future.result()
                num2 += 1
                print("get executor {} success".format(num2))
        mean_regret_vec = np.mean(total_regret_vec, axis=0)
        std_regret_vec = np.std(total_regret_vec, axis=0) / math.sqrt(num_simu) # error bar
        #print("k = {}, mean regret is: {}".format(k, mean_regret_vec[-1]))
        pd.DataFrame(mean_regret_vec).to_csv("./mean_value_T_{}_k_{}_d_{}_linTS_ind{}.csv".format(T, k, d, indi), header=None, index=False)
        pd.DataFrame(std_regret_vec).to_csv("./std_value_T_{}_k_{}_d_{}_linTS_ind{}.csv".format(T, k, d, indi), header=None, index=False)
            #draw(mean_regret_vec, std_regret_vec, T, k, p, case)
    #draw(mean_regret_vec, std_regret_vec, T, k, p, case)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=10000, help="number of time horizon")
    # parser.add_argument('--k', type=int, default=2, help="number of arms")
    parser.add_argument('--d', type=int, default=4, help="number of covairates(with intercept)")
    parser.add_argument('--R', type=float, default = 0.1, help="R-sub-Gaussian constant")
    parser.add_argument('--delta', type=float, default=1, help="the probability that confidence intervals fail")
    parser.add_argument('--sigma_e', type=float, default = 0.5, help="sigma value for gaussian distribution")
    parser.add_argument('--xmax', type=int, default=1, help="for truncated value from context distribution")
    parser.add_argument('--arm_para_noise', type=int, default=0, help="whether to use correct noise")
    parser.add_argument('--case', type=int, default = 1, help="which arm paramater to use")
    parser.add_argument('--indi', type=int, default = 1, help="split run 1000 rounds indicator")
    opt = parser.parse_args()
    return opt


def main(opt):
    simulate(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)



# if __name__ == "__main__":
#     k = 2
#     d = 4
#     delta = 0.99
#     R = 0.2
#     np.random.seed(41)
#     torch.manual_seed(41)
#     # arm_para = np.random.multivariate_normal(-np.ones(d), np.eye(d), k)
#     # context, opt_arm = generate_context_opt_arm(10, k, d, arm_para)
#     # print(opt_arm[1])
#     # context_array = np.asarray([context[1][action_id]
#     #                         for action_id in range(k)])
#     regret_vec = LinTS(T = 10000, k = k, d = d, R = R, delta = delta, sigma_e = 0.5, xmax = 1, arm_para_noise = 0, case = 1)

#     print(regret_vec[-1])     
        
        
        
        
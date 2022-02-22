#import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from concurrent.futures import ThreadPoolExecutor,as_completed,wait,ALL_COMPLETED
import threading
import argparse

def oful_with_exploite(T, kappa1, k, d, lmd, m2, sigma_e, xmax, arm_para_noise, switch, case):
    '''
    @params:
    T: time horizon
    rounds: number of rounds to perform OFUL
    k: number of arm
    d: number of covairiates with intercept
    lmd: ridge regression parameter lmd
    m2: upper bound of context
    sigma_e: R-subgaussian constant
    xmax: upper bound of the context values
    arm_para_noise: whether use context generated from random normal or from multivariate normal
    switch: binary value, wheter to switch to pure exloite algorithm (run OFUL if switch = 0 else run Pure Exploite)
    
    return: regret vector
    '''
    delta = float(1/T)
    rounds = int(math.ceil(k*d*((math.log(T)) ** kappa1)))
    VtInv = np.eye(k*d)/ lmd # (k*d, k*d) matrix
    # 1 0 0 0 0 0
    # 0 1 0 0 0 0 
    # 0 0 1 0 0 0
    # 0 0 0 1 0 0
    # 0 0 0 0 1 0
    # 0 0 0 0 0 1

    logVtDet = [d*math.log(lmd)] * k
    # arm parameter
    # generate from a distribution / fix -> ref: mostly exploration free
    if arm_para_noise == 1:
        arm_para = np.random.normal(0,1, size = (k, d)) # k*d matrix
    else:
        if case == 1: # same as stated in Bastani et al. mostly exploration free, mixture
            indicator = np.random.random()
            if indicator > 0.5:
                arm_para = np.random.multivariate_normal(-np.ones(d), np.eye(d), k)
            else:
                arm_para = np.random.multivariate_normal(np.ones(d), np.eye(d), k)
        elif case == 2:
            arm_para = np.random.multivariate_normal(np.ones(d), np.eye(d), k) # case 2: ~N(1d, 1d)
        elif case == 3:
            arm_para = np.random.multivariate_normal(np.zeros(d), np.eye(d), k) # case 3: ~N(0d, 1d)

    
    # arm estimation, k*d matrix
    theta = np.zeros((k, d))
    # reward per rounk for each arm, T*k matrix
    reward_vec = [[0] * k for _ in range(T)]
    # arm that pulled in round t in [1, ..., T], only values 0,1 are accepted
    optimal_ind = [-1] * T
    # regret vector in round t in [1, ..., T], 1*T matrix
    regret_vec = [0] * T # cumulative regret
    # current value of XtYt
    XtYt = np.zeros((k*d, 1))
    # error, follows standard normal distribution
    e = np.random.normal(0, 1, size = (1, T)) * sigma_e
    # each arm has its own radius
    radius_t = [0] * k
    
    #print("initialization finished")

    t = 0
    while t < rounds and switch == 0:
        ######################  Run LinUCB algorithm ##########################
        X_ = np.random.multivariate_normal(np.zeros((d)), np.eye(d), 1) # (1*d) context
        X = np.clip(X_, -xmax, xmax)
        #X = X_ / np.sqrt(np.sum(X_**2))
        X[0][0] = 1 # intercept
        # reshape to (d*1)
        x = np.transpose(X)
        # calculate log determinant, each k has a unique radius
        for i in range(k):
            radius_t[i] = math.sqrt(lmd) * m2 + sigma_e * math.sqrt(2*math.log(1/delta) + (logVtDet[i] - d * math.log(lmd)))
        
        # vector to store tmp reward
        tmp_reward_counter = np.zeros((k,1))

        # compute reward for each arm
        for i in range(k):
            tmp_reward_counter[i] = math.sqrt(np.transpose(x) @ VtInv[(i)*d:((i+1)*d), (i)*d:((i+1)*d)] @ x)
        
        # compute theta @ x first, and then add the bonus part for each arm
        op_reward = theta @ x # k*d d*1 -> k*1
        for i in range(len(radius_t)):
            op_reward[i] += math.sqrt(radius_t[i]) * (tmp_reward_counter[i])

        if t == 0: # randomly select
            op_arm = np.random.randint(0, len(op_reward.tolist()))
        else:
            # return index of optimal arm
            op_arm = np.argmax(op_reward)
        
        # # record which arm been pulled
        optimal_ind[t] = int(op_arm)

        # calculate reward based on the arm pulled
        theo_reward = arm_para @ x # rewards for each arm in theory
        round_reward = theo_reward[op_arm]
        true_reward = max(theo_reward)
        # calculate cumulative regret
        if t < 1:
            regret_vec[t] = float(true_reward - round_reward)
        else:
            regret_vec[t] = float(regret_vec[t-1]) + float(true_reward - round_reward)

        # record reward
        reward_vec[t][op_arm] = float(round_reward + e[0][t])

        # update inverse matrix using woodbury formula
    
        vtxtxtvt = VtInv[op_arm*d:((op_arm+1)*d), (op_arm)*d:((op_arm+1)*d)] @ x @ np.transpose(x) @ VtInv[op_arm*d:((op_arm+1)*d), (op_arm)*d:((op_arm+1)*d)]
        deno = (1 + float(np.transpose(x) @ VtInv[op_arm*d:((op_arm+1)*d), (op_arm)*d:((op_arm+1)*d)] @ x))
        VtInv[op_arm*d:((op_arm+1)*d), (op_arm)*d:((op_arm+1)*d)] -= (vtxtxtvt / deno)
        
        # update log determinant
        
        logVtDet[op_arm] += math.log((1 + (np.transpose(x) @ VtInv[(op_arm)*d:((op_arm+1)*d), (op_arm)*d:((op_arm+1)*d)] @ x)))

        # update the sum (XtYt)
        XtYt[op_arm*d:((op_arm+1)*d)] = XtYt[op_arm*d:((op_arm+1)*d)] + (x * reward_vec[t][op_arm])

        # update estimator theta
        theta_new = VtInv[op_arm*d:((op_arm+1)*d), (op_arm)*d:((op_arm+1)*d)] @ XtYt[op_arm*d:((op_arm+1)*d)] # d*1
        theta[op_arm,:] = np.transpose(theta_new) # 1*d

        t += 1
    
    changet = t
    switch = 1

    ###################################### Run Pure Exploit Algotithm ######################################
    for t in range(changet, T):
        X_ = np.random.multivariate_normal(np.zeros((d)), np.eye(d), 1) # (1*d) context
        X = np.clip(X_, -xmax, xmax)
        #X = X_ / np.sqrt(np.sum(X_**2))
        X[0][0] = 1 # intercept
        # reshape to (d*1)
        x = np.transpose(X)
        # vector to store tmp reward
        # tmp_reward_counter = np.zeros((k,1))
        # for i in range(k):
        #     tmp_reward_counter[i] = math.sqrt(np.transpose(x) @ VtInv[(i)*d:((i+1)*d), (i)*d:((i+1)*d)] @ x)
        
        op_reward = theta @ x # no upper bound in pure exploite

        # return index of optimal arm
        op_arm = np.argmax(op_reward)
        # # record which arm been pulled
        optimal_ind[t] = int(op_arm)

        # calculate reward based on the arm pulled
        theo_reward = arm_para @ x # rewards for each arm in theory
        round_reward = theo_reward[op_arm]
        true_reward = max(theo_reward)
        # calculate cumulative regret
        if t < 1:
            regret_vec[t] = float(true_reward - round_reward)
        else:
            regret_vec[t] = float(regret_vec[t-1]) + float(true_reward - round_reward)

        # record reward
        reward_vec[t][op_arm] = float(round_reward + e[0][t])

        # update inverse matrix using woodbury formula
        # tmp_context = np.zeros((k*d, 1))
        # tmp_context[(op_arm * d): (op_arm+1)*d] = x
    
        vtxtxtvt = VtInv[op_arm*d:((op_arm+1)*d), (op_arm)*d:((op_arm+1)*d)] @ x @ np.transpose(x) @ VtInv[op_arm*d:((op_arm+1)*d), (op_arm)*d:((op_arm+1)*d)]
        deno = (1 + float(np.transpose(x) @ VtInv[op_arm*d:((op_arm+1)*d), (op_arm)*d:((op_arm+1)*d)] @ x))
        VtInv[op_arm*d:((op_arm+1)*d), (op_arm)*d:((op_arm+1)*d)] -= (vtxtxtvt / deno)

        # update the sum (XtYt)
        XtYt[op_arm*d:((op_arm+1)*d)] = XtYt[op_arm*d:((op_arm+1)*d)] + (x * reward_vec[t][op_arm])

        # update estimator theta
        theta_new = VtInv[op_arm*d:((op_arm+1)*d), (op_arm)*d:((op_arm+1)*d)] @ XtYt[op_arm*d:((op_arm+1)*d)] # d*1
        theta[op_arm,:] = np.transpose(theta_new) # 1*d
    
    return regret_vec

def draw(rsltMean, rsltStd, T, k, p, case):
    plt.figure(figsize=(10,10))
    x = list(range(len(rsltMean)))
    y = rsltMean
    #plt.plot(x, y, linestyle = "--", label = "OFUL")
    plt.errorbar(x,y,yerr=rsltStd,fmt='-',ecolor='r',color='b',elinewidth=1,capsize=4, label = "Tr-LinUCB")
    plt.title("Cumulative Regret in {} round, kappa = {}, independent radius, max regret is: {}".format(T, p, round(max(rsltMean), 3)))
    plt.xlabel("Round")
    plt.ylabel("Regret")
    #plt.ylim(0, 120)
    plt.legend(loc = "lower right")
    plt.savefig("./figures/kappa_{}_T_{}_case_{}.jpg".format(p, T, case))

def runOnce(total_regret_vec, T, kappa1, k, d, lmd, m2, sigma_e, xmax, arm_para_noise, switch, case, i):
    lock = threading.Lock()
    #total_regret_vec = np.zeros((1000, T))
    temp_reg = oful_with_exploite(T, kappa1, k, d, lmd, m2, sigma_e, xmax, arm_para_noise, switch, case)
    # when write data in the array, a thread lock is required
    lock.acquire()
    total_regret_vec[i,:] = temp_reg
    lock.release()


def simulate(T, kappa, k, d, lmd, m2, sigma_e, xmax, arm_para_noise, switch, case):
    np.random.seed(41)
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
    num2 = 0
    num_simu = 1000
    #total_regret_vec = np.zeros((num_simu, T))
    #ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=10))
    ncpus = 10
    total_regret_vec = np.zeros((num_simu, T))
    # enable multi-thread
    with ThreadPoolExecutor(max_workers=ncpus) as executor:
        task_all = [executor.submit(runOnce, total_regret_vec, T, kappa, k, d, lmd, m2, sigma_e, xmax, arm_para_noise, switch, case, i) for i in range(num_simu)]
        for future in as_completed(task_all):
            data = future.result()
            num2 += 1
            print("get executor {} success".format(num2))
    mean_regret_vec = np.mean(total_regret_vec, axis=0)
    std_regret_vec = np.std(total_regret_vec, axis=0) / math.sqrt(num_simu) # error bar
    pd.DataFrame(mean_regret_vec).to_csv("./mean_value_T_{}_case_{}_ka_{}_trlucb_norm.csv".format(T, case, kappa), header=None, index=False)
    pd.DataFrame(std_regret_vec).to_csv("./std_value_T_{}_case_{}_ka_{}_trlucb_norm.csv".format(T, case, kappa), header=None, index=False)
    #draw(mean_regret_vec, std_regret_vec, T, k, p, case)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=10000, help="number of time horizon")
    parser.add_argument('--kappa', type=float, default = 2.0, help="power for OFUL rounds")
    parser.add_argument('--k', type=int, default=2, help="number of arms")
    parser.add_argument('--d', type=int, default=4, help="number of covairates(with intercept)")
    parser.add_argument('--lmd', type=float, default = 0.1, help="lambda value")
    parser.add_argument('--m2', type=int, default=1, help="upper bound of parameter")
    parser.add_argument('--sigma_e', type=float, default = 0.5, help="sigma value for gaussian distribution")
    parser.add_argument('--xmax', type=int, default=1, help="for truncated value from context distribution")
    parser.add_argument('--arm_para_noise', type=int, default=0, help="whether to use correct noise")
    parser.add_argument('--switch', type=int, default=0, help="indicator to switch to pure exploit from OFUL")
    parser.add_argument('--case', type=int, default = 1, help="which arm paramater to use")
    opt = parser.parse_args()
    return opt


def main(opt):
    simulate(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# run simluation experiment on real-world data

import math
import time
import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from get_data import *
from tr_linucb import *
from linTS_torch import *
from oful import *
from configs.dataset_config import META_CONFIGS

def get_theo_reward(T, k, y_df):
    
    reward = np.zeros((T, k))
    for i in range(T):
        reward[i, int(y_df[i])] = 1
    
    return reward

def normalize_context(X):
    # normalize context in each time step
    for i in range(X.shape[0]):
        X[i,:] = X[i,:] / np.linalg.norm(X[i,:], ord=2)
    
    return X

def convert_to_tensor(X, Y):
    X = torch.from_numpy(X).type(torch.float64)
    Y = torch.from_numpy(Y).type(torch.float64)
    return X, Y.unsqueeze(1)


def run_simulation_once(algo, X, Y, config):
    X = normalize_context(X)
    if algo == "trlucb" or algo == "oful":
        T = X.shape[0]
        k = np.unique(Y).shape[0]
        d = X.shape[1]
    elif algo == "linTS":
        X, Y = convert_to_tensor(X, Y)
        T = X.shape[0]
        k = torch.unique(Y).shape[0]
        d = X.shape[1]
    else:
        raise ValueError("Algorithm not implemented! For GF, OLS, run matlab code.")
    
    #print(T,k,d)
    # if algo == "trlucb":
    #     policy = oful_with_exploite
    # elif algo == "linTS":
    #     policy = LinTS
    # elif algo == "oful":
    #     policy = oful
    theo_reward = get_theo_reward(T, k, Y)
    
    if algo == "linTS":
        theo_reward = torch.from_numpy(theo_reward).type(torch.float64)
    
    # random permute patients
    rand_perm_ind = np.random.permutation(T)
    X = X[rand_perm_ind, :]
    Y = Y[rand_perm_ind]
    theo_reward = theo_reward[rand_perm_ind, :]
    
    # some findings about configs for different dataset:
    # cardio: kappa1 = 1.3, lmd = 0.0000001, m2 = 1, sigma_e = 1, xmax = 1, lmd = 0.01 is a cut off point
    # eeg: kappa1 = 2.5, lmd = 0.0001/0.0000001, m2 = 1, sigma_e = 1, xmax = 1, lmd = 0.0001 is a cut off point
    # eye_movements kappa1 = 1.8, lmd = 0.0000001, m2 = 1, sigma_e = 1, xmax = 1, lmd = 0.0000001 is a cut off point
    # warfine: kappa1 = 1.1, lmd = 0.0000001, m2 = 1, sigma_e = 1, xmax = 1, lmd = 0.0000001 is a cut off point
    #regret, pulled_arm = policy(T, kappa1 = 1.1, k=k, d=d, lmd=0.0000001, m2=1, sigma_e=1, xmax=1, theo_reward=theo_reward, context=X) trlinucb
    #regret, pulled_arm = policy(T, k=k, d=d, lmd=trl_oful_lmd, m2=1, sigma_e=1, theo_reward=theo_reward, context=X) # oful
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    
    sigma_e = config[algo]["sigma_e"]
    
    if algo == "trlucb":
        kappa = config[algo]["kappa1"]
        lmd = config[algo]["lmd"]
        m2 = config[algo]["m2"]
        regret, pulled_arm = oful_with_exploite(T, kappa1 = kappa, k=k, d=d, lmd=lmd, m2=m2, sigma_e=sigma_e, theo_reward=theo_reward, context=X)
    elif algo == "oful":
        lmd = config[algo]["lmd"]
        m2 = config[algo]["m2"]
        regret, pulled_arm = oful(T, k=k, d=d, lmd=lmd, m2=1, sigma_e=sigma_e, theo_reward=theo_reward, context=X)
    elif algo == "linTS":
        r = config[algo]["r"]
        delta = config[algo]["delta"]
        regret, pulled_arm = LinTS(T, k=k, d=d, R=r, delta=delta, sigma_e=sigma_e, theo_reward=theo_reward, context=X, device=device)
    else:
        raise NotImplementedError("This algorithm is not implemented yet! For GF, OLS, run matlab code of the mostly exploreation-free paper.")
    #print(regret[-1])
    
    fractions = np.mean(pulled_arm, axis=0) # fraction of pulling each arm
    
    # misclassified = 0
    # fraction_of_misclassified = 0
    # for i in range(len(pulled_arm)):
    #     if pulled_arm[i] != Y[i]:
    #         misclassified += 1
    # fraction_of_misclassified = misclassified / len(pulled_arm) 
    
    
    #print(fraction_of_misclassified)
    
    return regret, fractions

if __name__ == "__main__":
    np.random.seed(41)
    torch.random.manual_seed(41)
    #dataset = "./datasets/cardiotocography.csv"
    for file in os.listdir("./datasets"):
        dataset_name = file.replace(".csv", "")
        print("Dataset: ", dataset_name)
        dataset_dir = os.path.join("./datasets/{}.csv".format(dataset_name))
        X, Y = process_data(dataset_dir)
        print(X.shape, Y.shape)
        #print(X[:10,:])
        #print(np.unique(Y))
        ns = 100 # num simulation
        # trl_oful_lmd = 1e-6 #0.000001
        # linTS_r = 0.1
        # linTS_delta = 0.99
        dat_spec_config = META_CONFIGS[dataset_name]
        # run simulation once
        algo = "trlucb"
        print("algo: ", algo)
        # regret_counter = []
        # misclassified_counter = []
        total_regret = np.zeros((ns, X.shape[0]))
        
        for i in range(ns):
            start = time.time()
            regret, fraction_of_misclassified = run_simulation_once(algo, X, Y, dat_spec_config)
            total_regret[i,:] = regret
            #regret_counter.append(regret[-1])
            #misclassified_counter.append(fraction_of_misclassified)
            #print(regret[-1], fraction_of_misclassified)
            print("time used for {}th experiment: {}".format(i, time.time()-start))
        
        # write to csv file
        #df = pd.DataFrame({"regret": regret_counter, "misclassified": misclassified_counter})
        target_dir = "./results"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        mean_regret_vec = np.mean(total_regret, axis=0)
        std_regret_vec = np.std(total_regret, axis=0) / math.sqrt(ns) # error bar
        if algo == "linTS":
            r = dat_spec_config[algo]["r"]
            delta = dat_spec_config[algo]["delta"]
            pd.DataFrame(mean_regret_vec).to_csv(os.path.join(target_dir, "mean_{}_{}_r_{}_delta_{}_results.csv".format(algo, dataset_name, r, delta)), index=False)
        elif algo == "oful":
            lmd = dat_spec_config[algo]["lmd"]
            pd.DataFrame(mean_regret_vec).to_csv(os.path.join(target_dir, "mean_{}_{}_lmd_{}_results.csv".format(algo, dataset_name, lmd)), index=False)
        elif algo == "trlucb":
            lmd = dat_spec_config[algo]["lmd"]
            pd.DataFrame(mean_regret_vec).to_csv(os.path.join(target_dir, "mean_{}_{}_lmd_{}_results.csv".format(algo, dataset_name, lmd)), index=False)
        # # plot regret
        # plt.plot(regret)
        # plt.xlabel("Time")
        # plt.ylabel("Regret")
        # plt.title("Regret of {}".format(algo))
        # plt.show()
        
        # # plot fraction of misclassified
        # plt.plot(fraction_of_misclassified)
        # plt.xlabel("Time")
        # plt.ylabel("Fraction of misclassified")
        # plt.title("Fraction of misclassified of {}".format(algo))
        # plt.show()
    
    
    
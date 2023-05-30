
# preprocess desired dataset

import torch
import numpy as np
import pandas as pd
import os
import sys

sys.path.append("../")

def process_data(dataset):
    # read csv file
    temp = pd.read_csv(dataset, header=None)
    # remove first row (header)
    df = temp.iloc[1:]
    # convert to numpy array
    all_df = df.to_numpy()
    # seperate contexts and labels
    X = all_df[:, :-1].astype(np.float64)
    Y = all_df[:, -1].astype(np.float64)
    
    # change all label from 1-start to 0-start to avoid index error
    if np.min(Y) != 0:
        Y = Y - 1
 
    #Y = Y - np.min(Y) + 1 # because this is the number of arms K
    #print(X[:5,:], Y[:5])
    return X, Y

# if __name__ == "__main__":
#     dataset = "./datasets/cardiotocography.csv"
#     X, Y = process_data(dataset)
#     print(X.shape, Y.shape)
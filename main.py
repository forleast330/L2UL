"""
codes for L2UL.
Take LR as an example.
"""

# Authors: L2UL

import numpy as np 
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from preprocessing import preprocessing
from unlearn import unlearn


def main():
    fea, gt = load_data("data")   # load the data
    psi, t = 4, 100               # parameter setting
    net, data_map, mean_map_ = preprocessing(psi, t, fea, gt)

    # Train A(D)
    #------------------------------------------------------------------------------------------------
    seed4un_split = 2023
    indices = np.arange(fea.shape[0]) 
    indices_train, indices_test, y_train, y_test = train_test_split(indices, gt, test_size=0.2, random_state=seed4un_split, stratify=gt)
    X_train, X_test = fea[indices_train,:], fea[indices_test,:]
    clf = LogisticRegressionCV(cv=5, max_iter=500, random_state=2023, scoring="accuracy").fit(X_train, y_train)   
    print(clf.score(X_test, y_test))
    #----------------------------------end----------------------------------

    # D_f
    #------------------------------------------------------------------------------------------------
    index4un = [i for i in range(1)]
    indices = np.arange(X_train.shape[0])
    retrain_set = indices[list(range(0, min(index4un))) +list(range(max(index4un)+1, X_train.shape[0]))]
    #----------------------------------end----------------------------------

    # Baseline Retrain A(D\D_f)
    #------------------------------------------------------------------------------------------------
    clf_re = LogisticRegressionCV(cv=5, max_iter=500, random_state=2023, scoring="accuracy").fit(X_train[retrain_set,:], y_train[retrain_set]) 
    print(clf_re.score(X_test, y_test))
    #----------------------------------end----------------------------------


    # unlearn D_f using trained U
    #------------------------------------------------------------------------------------------------  
    mean_map_un = np.mean(data_map[indices_train[index4un],:], axis=0)
    clf_un = unlearn(net, mean_map_, mean_map_un, clf)
    print(clf_un.score(X_test, y_test))
    #----------------------------------end----------------------------------
if __name__ == '__main__':
    main()

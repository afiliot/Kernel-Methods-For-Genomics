import utils
from SVM import C_SVM
import ALIGNF
import NLCKernels
import numpy as np
from itertools import product
from tqdm import tqdm
import pickle as pkl
import os
import pandas as pd


# What to do ?

"""
    ******************************************* Kernel methods available ************************************************
    - SP_k{x} : Spectrum kernel with x = int
    - WD_d{x} : Weight degree kernel with x = int
    - WDS_d{x}_s{y} : Weight degree kernel with shifts with x = int (d), y = int (s)
    - MM_k{x}_m{y}: Mismatch kernel with x = int (k) and y = int (m)
    - LA_e{x}_d{y}_b{z}_smith{X}_eig{Y} with x = int (e), y = int (d), z = float (beta), X = 0/1 (smith), Y = 0/1 (eig)
    - SS_l{x}_k{y} with x = float (lambda), y = int (k)
    - GP_k{x}_g{y} with x = int (k), y = int (gap g)
    ********************************************************************************************************************
"""

build_kernel = False  # Build a kernel
check_method = False  # Use a particular method
check_alignf = False  # Use ALIGNF algorithm
check_NLCK   = False  # Use NLCK algorithm
check_CVNLCK = False  # Use cross validation on NLCK hyperparameters
check_other  = False  # Free

if __name__ == '__main__':
    if build_kernel:
        methods = ['GP_k3_g1', 'MM_k5_m1', 'WD_d10']
        for method in methods:
            X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method=method, replace=True)
            # Put replace = False not to erase the previous saves

    elif check_method:
        method = 'MM_k6_m1'
        algo = 'CSVM'
        solver = 'CVX'
        data, data1, data2, data3, K, ID = utils.get_all_data([method])
        Cs = np.sort([i * 10 ** j for (i, j) in product(range(1, 10), range(-2, 1))])
        # Perform cross validation on data set 1 (TF = 1)
        utils.cross_validation(Ps=Cs, data=data1, algo=algo, solver=solver, kfolds=3, K=K, ID=ID)

    elif check_alignf:
        methods = ['MM_k3_m1', 'WD_d5', 'SS_l1_k3']
        data, data1, data2, data3, kernels, ID = ALIGNF.aligned_kernels(methods)
        K = kernels[0]  # 0 index for first data set
        X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, K_1, ID_1 = utils.reformat_data(data1, [K], ID)
        Cs = np.sort([i * 10 ** j for (i, j) in product(range(1, 10), range(-3, 2))])
        utils.cross_validation(Ps=Cs, data=data1, algo='CSVM', kfolds=5, K=K_1[0], ID=ID_1)

    elif check_NLCK:
        methods = ['SP_k6', 'SP_k5', 'SP_k4']
        data, data1, data2, data3, kernels, ID = utils.get_all_data(methods)
        X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, kernels_1, ID_1 = utils.reformat_data(data1, kernels, ID)
        Km1 = NLCKernels.NLCK(X_train_1, y_train_1, ID_1, kernels_1, C=1e-2, eps=1e-9, degree=2).get_K()
        Cs = np.sort([i * 10 ** j for (i, j) in product(range(1, 10), range(-3, 5))])
        utils.cross_validation(Ps=Cs, data=data1, algo='CSVM', kfolds=5, K=Km1, ID=ID_1)

    elif check_CVNLCK:
        methods = ['MM_k3_m1', 'WD_k5', 'SS_l1_k3']
        Cs_NLK = [1e-3, 1e-2, 0.1, 1, 10, 100]
        Cs_SVM = np.concatenate((np.linspace(0.01, 0.1, 19), np.linspace(0.1, 1, 91), np.linspace(1, 10, 19)))
        degrees = [1]
        lbdas = [1, 5, 10, 50, 100]
        NLCKernels.cross_validation(k=1, methods=methods, Cs_NLK=Cs_NLK, Cs_SVM=Cs_SVM, degrees=degrees, lambdas=lbdas)

    elif check_other:
        pass













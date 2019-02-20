import pandas as pd
import numpy as np
import os
import warnings
import kernels as km
import pickle as pkl
import datetime
import utils
from tqdm import tqdm
from SVM import C_SVM
from KLR import KLR
from KRR import KRR
import operator
from itertools import product


def get_train(k):
    """
    Load training data set specified by k. Replace 0s by -1s in target and insert a flag k.
    :param k: int, which data set to load
    :return:
        - X: pd.DataFrame, features
        - y: pd.DataFrame, labels (0/1)
    """
    X = pd.read_csv('./Data/Xtr' + str(k) + '.csv')
    y = pd.read_csv('./Data/Ytr' + str(k) + '.csv')
    y['Bound'] = y['Bound'].replace(0, -1)
    X.insert(1, 'k', k+1)
    y.insert(1, 'k', k+1)
    return X, y


def get_test(k):
    """
    Load testing data set specified by k
    :param k: int, which data set to load
    :return:
        - X: pd.DataFrame, features
    """
    X = pd.read_csv('./Data/Xte' + str(k) + '.csv')
    X.insert(1, 'k', k+1)
    return X


def train_test_split_k(X, y, p):
    """
    Split training data into training (p%) and testing set (1-p %), with usually p=0.75.
    Training and testing data sets are balanced w.r.t to the target y.
    :param X: pd.DataFrame, features
    :param y: pd.DataFrame, labels (0/1)
    :return:
        - X_train: pd.DataFrame, training features
        - X_val: pd.DataFrame, testing features
        - y_train: pd.DataFrame, training labels
        - y_val: pd.DataFrame, testing labels
    """
    idx_0, idx_1 = np.where(y.loc[:, "Bound"] == -1)[0], np.where(y.loc[:, "Bound"] == 1)[0]
    n0, n1 = len(idx_0), len(idx_1)
    idx_tr0, idx_tr1 = idx_0[:int(p * n0)+1], idx_1[:int(p * n1)+1]
    idx_te0, idx_te1 = list(set(idx_0) - set(idx_tr0)), list(set(idx_1) - set(idx_tr1))
    idx_tr, idx_te = np.concatenate((idx_tr0, idx_tr1)), np.concatenate((idx_te0, idx_te1))
    X_train, y_train = X.iloc[idx_tr, :], y.iloc[idx_tr, :]
    X_val, y_val = X.iloc[idx_te, :], y.iloc[idx_te, :]
    return X_train, y_train, X_val, y_val


def trainInRepo(file):
    """
    Check if training data have already been stored in the repo
    :param file: string, file name
    :return Boolean
    """
    return file in os.listdir('./Data')


def resetIndex(df):
    """
    Reset index of pd.dataFrames
    :param df: list of pd.dataFrame
    :return: list of pd.dataFrame
    """
    D = []
    for d in df:
        d.reset_index(drop=True, inplace=True)
        D.append(d)
    return D


def train_test_split():
    """
    Split training data into training (p%) and testing set (1-p %), with usually p=0.75
    for each of the 3 types of TF and concatenate trains and vals.
    Training and testing data sets are balanced w.r.t to the target y.
    :return:
        - X_train: pd.DataFrame, training features
        - X_val: pd.DataFrame, validation features
        - y_train: pd.DataFrame, training labels
        - y_val: pd.DataFrame, validation labels
        - X_test: pd.DataFrame, testing features
    """
    for k in range(3):
        X, y = get_train(k)
        if k == 0:
            X_train, y_train, X_val, y_val = train_test_split_k(X, y, 0.75)
            X_test = get_test(k)
        else:
            X_tr, y_tr, X_v, y_te = train_test_split_k(X, y, 0.75)
            X_train, X_val = pd.concat((X_train, X_tr), axis=0), pd.concat((X_val, X_v), axis=0)
            y_train, y_val = pd.concat((y_train, y_tr), axis=0), pd.concat((y_val, y_te), axis=0)
            X_test = pd.concat((X_test, get_test(k)), axis=0)
        X_train, X_val, y_train, y_val, X_test = resetIndex([X_train, X_val, y_train, y_val, X_test])
    return X_train, y_train, X_val, y_val, X_test


def get_training_datas(method, all=True, replace=False):
    """
    Construct training and testing data, and kernels.
    :param: d: int, maximal degree for Weighted Degree Kernel
    :param: method: string, method used for computing kernels
    :param: replace: Boolean, whether or not replace the existing files in the repo
    :return:
        - X_train: pd.DataFrame, training features
        - X_val: pd.DataFrame, validation features
        - y_train: pd.DataFrame, training labels
        - y_val: pd.DataFrame, validation labels
        - X_test: pd.DataFrame, testing features
        - K: np.array, kernel
        - ID: np.array, Ids
    """
    warnings.filterwarnings('ignore')
    file = 'training_data_'+method+'.pkl'
    if not all:
        X_train, y_train, X_val, y_val, X_test = train_test_split()
        X_test.loc[:, 'Id'] = -(X_test.loc[:, 'Id'] + 1)
        X = pd.concat((X_train, X_val, X_test), axis=0)
        ID = X.loc[:, 'Id']
    else:
        if trainInRepo(file) and not replace:
            X_train, y_train, X_val, y_val, X_test, K, ID = pkl.load(open(os.path.join('./Data', file), 'rb'))
        else:
            X_train, y_train, X_val, y_val, X_test = train_test_split()
            X_test.loc[:, 'Id'] = -(X_test.loc[:, 'Id']+1)
            X = pd.concat((X_train, X_val, X_test), axis=0)
            ID = np.array(X.loc[:, 'Id'])
            K = km.select_method(X, method)
            file = 'training_data_'+method+'.pkl'
            pkl.dump([X_train, y_train, X_val, y_val, X_test, K, ID], open(os.path.join('./Data', file), 'wb'))
    return X_train, y_train, X_val, y_val, X_test, K, ID


def select_k(k, X_train, y_train, X_val, y_val, X_test, K, ID):
    """
    Restrict training and testing data to 1 data set of interest, defined by k (TF type)
    :param k: int, which data set to restrict on
    :param X_train: pd.DataFrame, training features
    :param y_train: pd.DataFrame, training labels
    :param X_val: pd.DataFrame, validation features
    :param y_val: pd.DataFrame, validation labels
    :param X_test: pd.DataFrame, testing features
    :param K: np.array, kernel
    :param ID: np.array, IDs
    :return: pd.DataFrames and kernel
    """
    idx_train = np.where(np.array(X_train.loc[:, 'k']) == k)[0]
    idx_val = np.where(np.array(X_val.loc[:, 'k']) == k)[0]
    idx_test = np.where(np.array(X_test.loc[:, 'k']) == k)[0]
    id_train, id_val, id_test = X_train.iloc[idx_train, 0], X_val.iloc[idx_val, 0], X_test.iloc[idx_test, 0]
    id_k = np.concatenate((id_train, id_val, id_test))
    idx = np.where(np.in1d(ID, id_k))[0]
    X_train_, y_train_ = X_train.iloc[idx_train], y_train.iloc[idx_train]
    X_val_, y_val_ = X_val.iloc[idx_val], y_val.iloc[idx_val]
    X_test_ = X_test.iloc[idx_test]
    K_ = K[idx][:, idx]
    return X_train_, y_train_, X_val_, y_val_, X_test_, K_, id_k


def export_predictions(svms, X_tests):
    """
    Compute and export predictions on test set (0 or 1)
    :param svms: list, list of trained svms
    :param X_tests: list, list of testing pd.DataFrames
    :return: np.array, predictions
    """
    for k, svm in enumerate(svms):
        X_test = X_tests[k]
        pred_test = svm.predict(X_test).astype(int)
        if k == 0:
            y_test = pd.DataFrame({'Id': X_test.Id, 'Bound': pred_test})
        else:
            y_test = pd.concat((y_test, pd.DataFrame({'Id': X_test.Id, 'Bound': pred_test})))
    y_test.Id = -y_test.Id - 1
    y_test.Bound = y_test.Bound.replace(-1, 0)
    t = datetime.datetime.now().time()
    y_test.to_csv('y_test_' + str(t) + '.csv', index=False)
    return y_test


def sort_accuracies_k(algo='C_SVM', k=1):
    """
    Sort best accuracies obtained for each kernel methods (through cross validation) along with constants.
    :param algo: string, 'C_SVM' or 'SVM2'
    :param k: int, which data set to consider (default 1)
    :return: pd.DataFrame, methods+accuracies+Cs
    """
    k_ = 'k'+str(k)
    val_scores = {}
    C_opts = {}
    warnings.filterwarnings('ignore')
    for file in os.listdir('./Data/CrossVals/'):
        if algo in file and k_ in file:
            pred = pkl.load(open(os.path.join('./Data/CrossVals/', file), 'rb'))
            file = file.split(k_)[0][5:-1]
            val_scores[file] = np.max(pred[3])
            C_opts[file] = pred[4]
    sorted_val = sorted(val_scores.items(), key=operator.itemgetter(1), reverse=True)
    sorted_C = {}
    for i in range(len(sorted_val)):
        key = sorted_val[i][0]
        sorted_C[key] = C_opts[key]
    u = sorted_val, sorted_C
    p = pd.DataFrame({'Kernel Method - k='+str(k): [u[0][i][0] for i in range(len(u[0]))],
                      'Val accuracy - k='+str(k): [u[0][i][1] for i in range(len(u[0]))],
                      'Constant C - k='+str(k): [np.round(i, 4) for i in u[1].values()]})
    return p


def sort_accuracies(algo='C_SVM', k=3):
    """
    Sort accuracies for several data sets (if k=3, then all data sets will be considered)
    :param algo: string, 'C_SVM' or 'SVM2'
    :param k: int, which data set to consider (default 1)
    :return: pd.DataFrame, methods+accuracies+Cs
    """
    for k_ in range(1, k+1):
        if k_ == 1:
            p = sort_accuracies_k(algo, k_)
        else:
            p = pd.concat((p, sort_accuracies_k(algo, k_)), axis=1)
    return p


def load_kernel(method):
    _, _, _, _, _, K, _ = get_training_datas(method=method, replace=False)
    return K


def get_all_data(methods):
    """
    Return all the necessary data for training and running ALIGNF algorithm.
    :param methods: list of strings, kernel methods
    :return: list of data
    """
    X_train, y_train, X_val, y_val, X_test, K, ID = get_training_datas(method=methods[0], replace=False)
    X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, _, _ = select_k(1, X_train, y_train, X_val, y_val, X_test, K, ID)
    X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, _, _ = select_k(2, X_train, y_train, X_val, y_val, X_test, K, ID)
    X_train_3, y_train_3, X_val_3, y_val_3, X_test_3, _, _ = select_k(3, X_train, y_train, X_val, y_val, X_test, K, ID)
    data = [X_train, y_train, X_val, y_val, X_test]
    data1 = [X_train_1, y_train_1, X_val_1, y_val_1, X_test_1]
    data2 = [X_train_2, y_train_2, X_val_2, y_val_2, X_test_2]
    data3 = [X_train_3, y_train_3, X_val_3, y_val_3, X_test_3]
    kernels = []
    for m in methods:
        kernels.append(load_kernel(m))
    if len(kernels) == 1:
        kernels = kernels[0]
    return data, data1, data2, data3, kernels, ID


def export_predictions(method, algo, P_opts):
    """
    Export predictions for submission
    :param method: string, which kernel method to use
    :param algo: string, choose between CSVM, KLR or KRR
    :param P_opts: list, optimal values of constants for each data set (depending on the algorithm)
    :return: np.array, predictions
    """
    data, data1, data2, data3, K, ID = get_all_data([method])
    X_train_1, y_train_1, X_val_1, y_val_1, X_test_1 = data1
    X_train_2, y_train_2, X_val_2, y_val_2, X_test_2 = data2
    X_train_3, y_train_3, X_val_3, y_val_3, X_test_3 = data3
    P_opt_1, P_opt_2, P_opt_3 = P_opts
    if algo == 'CSVM':
        alg_1 = C_SVM(K=K, ID=ID, C=P_opt_1, print_callbacks=False)
        alg_2 = C_SVM(K=K, ID=ID, C=P_opt_2, print_callbacks=False)
        alg_3 = C_SVM(K=K, ID=ID, C=P_opt_3, print_callbacks=False)
    elif algo == 'KRR':
        alg_1 = KRR(K=K, ID=ID, lbda=P_opt_1)
        alg_2 = KRR(K=K, ID=ID, lbda=P_opt_2)
        alg_3 = KRR(K=K, ID=ID, lbda=P_opt_3)
    else:
        NotImplementedError('Please choose between "CSVM", "KRR" or "KLR"')
    alg_fit = []
    for alg, data in zip([alg_1, alg_2, alg_3], [data1, data2, data3]):
        X_train, y_train, X_val, y_val, X_test = data
        alg.fit(X_train, y_train)
        alg_fit.append(alg)
        pred_tr = alg.predict(X_train)
        print('Accuracy on train set: {:0.4f}'.format(alg.score(pred_tr, y_train)))
        pred_val = alg.predict(X_val)
        print('Accuracy on val set: {:0.4f}'.format(alg.score(pred_val, y_val_1)))
    y_pred_test = utils.export_predictions(alg_fit, [X_test_1, X_test_2, X_test_3])
    return y_pred_test


def cross_validation(Ps, data, algo, kfolds=5, pickleName='cv_C_SVM', **kwargs):
    """
    Cross-validation implementation for C_SVM
    :param Ps: list or np.array, list of constants to loop on
    :param data: list, X_train + y_train + X_val + y_val
    :param algo: string, choose between CSVM, KLR, KRR
    :param kfolds: int, number of folds used for CV
    :param pickleName: string
    :param kwargs: arguments used for C_SVM (tol, etc...)
    :return: list: - p_opt: float, optimal constant (best average score)
                   - scores_tr: np.array, scores on train set for each constant C (and each fold)
                   - scores_te: np.array, scores on val set for each constant C (and each fold)
                   - mean_scores_tr: np.array, average scores on train set for each constant C
                   - mean_scores_te: np.array, average scores on val set for each constant C
    """
    scores_tr = np.zeros((kfolds, len(Ps)))
    scores_te = np.zeros((kfolds, len(Ps)))
    X_tr, y_tr, X_te, y_te, _ = data
    X_train_ = pd.concat((X_tr, X_te)).reset_index(drop=True).sample(frac=1)
    y_train_ = pd.concat((y_tr, y_te)).reset_index(drop=True).iloc[X_train_.index]
    X_train_, y_train_ = X_train_.reset_index(drop=True), y_train_.reset_index(drop=True)
    n = X_train_.shape[0]
    p = int(n // kfolds)
    for k in tqdm(range(kfolds)):
        print('Fold {}'.format(k+1))
        q = p * (k + 1) + n % kfolds if k == kfolds - 1 else p * (k + 1)
        idx_val = np.arange(p * k, q)
        idx_train = np.setdiff1d(np.arange(n), idx_val)
        X_train, y_train = X_train_.iloc[idx_train, :], y_train_.iloc[idx_train, :]
        X_val, y_val = X_train_.iloc[idx_val, :], y_train_.iloc[idx_val, :]
        s_tr, s_te = [], []
        for P in Ps:
            if algo == 'CSVM':
                alg = C_SVM(C=P, print_callbacks=False, **kwargs)
            elif algo == 'KLR':
                alg = KLR(lbda=P, print_callbacks=False, **kwargs)
            elif algo == 'KRR':
                alg = KRR(lbda=P, **kwargs)
            else:
                NotImplementedError('Please choose between "CSVM", "KRR" or "KLR"')
            alg.fit(X_train, y_train)
            pred_tr = alg.predict(X_train)
            score_tr = alg.score(pred_tr, y_train)
            pred_te = alg.predict(X_val)
            score_te = alg.score(pred_te, y_val)
            s_tr.append(score_tr)
            s_te.append(score_te)
            print('Constant={}, train_acc={:0.4f}, val_acc={:0.4f}'.format(P, score_tr, score_te))
        scores_tr[k], scores_te[k] = s_tr, s_te
    mean_scores_tr, mean_scores_te = np.mean(scores_tr, axis=0), np.mean(scores_te, axis=0)
    p_opt = Ps[np.argmax(mean_scores_te)]
    print('Best constant={}, val_acc={:0.4f}'.format(p_opt, np.max(mean_scores_te)))
    pkl.dump([scores_tr, scores_te, mean_scores_tr, mean_scores_te, p_opt],
             open(os.path.join('./Data/CrossVals/', pickleName), 'wb'))
    return p_opt, scores_tr, scores_te, mean_scores_tr, mean_scores_te



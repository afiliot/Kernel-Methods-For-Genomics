import pandas as pd
import numpy as np
import os
import warnings
import kernels as km
import pickle as pkl
import datetime
from tqdm import tqdm
from SVM import C_SVM
from KLR import KLR
from KRR import KRR
import operator

path = '/Users/bfiliot/MVA/KERNEL/Data' #'./Data'


######################################Load raw training and testing data################################################


def get_train(k):
    """
    Load training data set specified by k. Replace 0 by -1 in target and insert a flag k.
    :param k: int, which data set to load
    :return:
        - X: pd.DataFrame, sequences
        - y: pd.DataFrame, labels (0/1)
    """
    X, y = pd.read_csv('./Data/Xtr' + str(k) + '.csv'), pd.read_csv('./Data/Ytr' + str(k) + '.csv')
    y['Bound'] = y['Bound'].replace(0, -1)
    X.insert(1, 'k', k+1)
    y.insert(1, 'k', k+1)
    return X, y


def get_test(k):
    """
    Load testing data set specified by k
    :param k: int, which data set to load
    :return:
        - X: pd.DataFrame, sequences
    """
    X = pd.read_csv('./Data/Xte' + str(k) + '.csv')
    X.insert(1, 'k', k+1)
    return X

###########################################Split train and test#########################################################


def train_test_split_k(X, y, p):
    """
    Split training data into training (p%) and testing set (1-p %), with usually p=0.75 for a single dataset.
    Training and testing data sets are balanced w.r.t to the target y.
    :param X: pd.DataFrame, sequences
    :param y: pd.DataFrame, labels (0/1)
    :param p: float between 0 and 1, split proportion
    :return:
        - X_train: pd.DataFrame, training sequences
        - X_val: pd.DataFrame, testing sequences
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


def train_test_split():
    """
    Split training data into training (p%) and testing set (1-p %), with usually p=0.75 for each of the 3 types
    of TF and concatenate trains and vals.
    Training and testing data sets are balanced w.r.t to the target y.
    :return:
        - X_train: pd.DataFrame, training sequences
        - X_val: pd.DataFrame, validation sequences
        - y_train: pd.DataFrame, training labels
        - y_val: pd.DataFrame, validation labels
        - X_test: pd.DataFrame, testing sequences
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


def select_k(k, X_train, y_train, X_val, y_val, X_test):
    """
    Restrict training and testing data to 1 data set of interest, defined by k (TF type)
    :param k: int, which data set to restrict on
    :param X_train: pd.DataFrame, training sequences
    :param y_train: pd.DataFrame, training labels
    :param X_val: pd.DataFrame, validation sequences
    :param y_val: pd.DataFrame, validation labels
    :param X_test: pd.DataFrame, testing sequences
    :return: pd.DataFrames:
        - X_train: training sequences
        - y_train: training labels
        - X_val: validation sequences
        - y_val: validation labels
        - X_test: testing sequences
    """
    idx_train = np.where(np.array(X_train.loc[:, 'k']) == k)[0]
    idx_val = np.where(np.array(X_val.loc[:, 'k']) == k)[0]
    idx_test = np.where(np.array(X_test.loc[:, 'k']) == k)[0]
    X_train_, y_train_ = X_train.iloc[idx_train], y_train.iloc[idx_train]
    X_val_, y_val_ = X_val.iloc[idx_val], y_val.iloc[idx_val]
    X_test_ = X_test.iloc[idx_test]
    return X_train_, y_train_, X_val_, y_val_, X_test_

#########################################Build kernels and load all data################################################


def get_training_datas(method, all=True, replace=False):
    """
    Construct training, testing data, and kernels.
    :param: method: string, method used for computing kernels
    :param: replace: Boolean, whether or not replace the existing files in the repo
    :return:
        - X_train: pd.DataFrame, training sequences
        - X_val: pd.DataFrame, validation sequences
        - y_train: pd.DataFrame, training labels
        - y_val: pd.DataFrame, validation labels
        - X_test: pd.DataFrame, testing sequences
        - K: np.array, kernel
        - ID: np.array, Ids
    """
    file = 'training_data_'+method+'.pkl'
    if not all:
        X_train, y_train, X_val, y_val, X_test = train_test_split()
        X_test.loc[:, 'Id'] = -(X_test.loc[:, 'Id'] + 1)
        X = pd.concat((X_train, X_val, X_test), axis=0)
        ID = X.loc[:, 'Id']
    else:
        if trainInRepo(file) and not replace:
            X_train, y_train, X_val, y_val, X_test, K, ID = pkl.load(open(os.path.join(path, file), 'rb'))
        else:
            X_train, y_train, X_val, y_val, X_test = train_test_split()
            X_test.loc[:, 'Id'] = -(X_test.loc[:, 'Id']+1)
            X = pd.concat((X_train, X_val, X_test), axis=0)
            ID = np.array(X.loc[:, 'Id'])
            K = km.select_method(X, method)
            file = 'training_data_'+method+'.pkl'
            pkl.dump([X_train, y_train, X_val, y_val, X_test, K, ID], open(os.path.join(path, file), 'wb'))
    return X_train, y_train, X_val, y_val, X_test, K, ID


def load_kernel(method):
    """
    Load kernel
    :param method: string, kernel method
    :return: np.array, kernel
    """
    _, _, _, _, _, K, _ = get_training_datas(method=method, replace=False)
    return K


def get_all_data(methods):
    """
    Return all the necessary data for training and running the experiments, for each TF type.
    Specially designed for ALIGNF and NLCK algorithms.
    :param methods: list of strings, kernel methods
    :return: list:
            - X_train, y_train, X_val, y_val, X_test
            - X_train_1, y_train_1, X_val_1, y_val_1, X_test_1
            - X_train_2, y_train_2, X_val_2, y_val_2, X_test_2
            - X_train_3, y_train_3, X_val_3, y_val_3, X_test_3
            - kernels (list of np.array kernels)
            - IDs (np.array)
    """
    print('Loading data...')
    X_train, y_train, X_val, y_val, X_test, K, ID = get_training_datas(method=methods[0], replace=False)
    X_train_1, y_train_1, X_val_1, y_val_1, X_test_1 = select_k(1, X_train, y_train, X_val, y_val, X_test)
    X_train_2, y_train_2, X_val_2, y_val_2, X_test_2 = select_k(2, X_train, y_train, X_val, y_val, X_test)
    X_train_3, y_train_3, X_val_3, y_val_3, X_test_3 = select_k(3, X_train, y_train, X_val, y_val, X_test)
    data = [X_train, y_train, X_val, y_val, X_test]
    data1 = [X_train_1, y_train_1, X_val_1, y_val_1, X_test_1]
    data2 = [X_train_2, y_train_2, X_val_2, y_val_2, X_test_2]
    data3 = [X_train_3, y_train_3, X_val_3, y_val_3, X_test_3]
    kernels = []
    for k, m in enumerate(methods):
        print('Kernel '+str(k+1)+'...')
        kernels.append(load_kernel(m))
    if len(kernels) == 1:
        kernels = kernels[0]
    return data, data1, data2, data3, kernels, ID

#########################################Cross validation generic method################################################


def cross_validation(Ps, data, algo, kfolds=5, **kwargs):
    """
    Cross-validation implementation for C_SVM, KLR or KRR
    :param Ps: list or np.array, list of constants to loop on
    :param data: list, X_train + y_train + X_val + y_val
    :param algo: string, choose between CSVM, KLR, KRR
    :param kfolds: int, number of folds used for CV
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
                alg = KLR(lbda=P, **kwargs)
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
    return p_opt, scores_tr, scores_te, mean_scores_tr, mean_scores_te

###################################################Predictions##########################################################


def export_predictions(algos, X_tests):
    """
    Compute and export predictions on test set (0 or 1)
    :param algos: list, list of trained svms
    :param X_tests: list, list of testing pd.DataFrames
    :return: np.array, predictions
    """
    for k, alg in enumerate(algos):
        X_test = X_tests[k]
        pred_test = alg.predict(X_test).astype(int)
        if k == 0:
            y_test = pd.DataFrame({'Id': X_test.Id, 'Bound': pred_test})
        else:
            y_test = pd.concat((y_test, pd.DataFrame({'Id': X_test.Id, 'Bound': pred_test})))
    y_test.Id = np.arange(1000 * len(algos))
    y_test.Bound = y_test.Bound.replace(-1, 0)
    t = datetime.datetime.now().time()
    y_test.to_csv('y_test_' + str(t) + '.csv', index=False)
    return y_test

#################################################Other functions########################################################


def reformat_data(data, kernels, ID):
    """
    Reformat data in order to make computations faster. Mostly useful for NLCK algorithm
    where reformat_data allows to compute the final non-linear combination only on the dataset of interest and not
    the whole union of the 3. This function formats the IDs.
    :param data: list (X_train, y_train, X_val, y_val, X_test)
    :param kernels: list of kernels
    :param ID: np.array, Ids
    :return: list:
            - X_train
            - y_train
            - X_val
            - y_val
            - X_test
            - kernels
            - ID
    """
    X_train, y_train, X_val, y_val, X_test = data
    ID_ = np.concatenate(
        (np.array(X_train.loc[:, 'Id']), np.array(X_val.loc[:, 'Id']), np.array(X_test.loc[:, 'Id'])))
    idx = np.array([np.where(ID == ID_[i])[0] for i in range(len(ID_))]).squeeze()
    kernels_ = []
    for K in tqdm(kernels):
        kernels_.append(K[idx][:, idx])
    ID_ = np.arange(ID_.shape[0])
    X_train.Id = ID_[:X_train.shape[0]]
    X_val.Id = ID_[X_train.shape[0]:(X_train.shape[0] + X_val.shape[0])]
    X_test.Id = ID_[(X_train.shape[0] + X_val.shape[0]):(
            X_train.shape[0] + X_val.shape[0] + X_test.shape[0])]
    y_train.Id = ID_[:y_train.shape[0]]
    y_val.Id = ID_[X_train.shape[0]:(X_train.shape[0] + X_val.shape[0])]
    return X_train, y_train, X_val, y_val, X_test, kernels_, ID_


def trainInRepo(file):
    """
    Check if training data have already been stored in your repository
    :param file: string, file name
    :return Boolean
    """
    return file in os.listdir(path)


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

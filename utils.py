import pandas as pd
import numpy as np
import os
import warnings
import kernels as km
import pickle as pkl


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
    return X


def train_test_split_k(X, y, p):
    """
    Split training data into training (p%) and testing set (1-p %), with usually p=0.75.
    Training and testing data sets are balanced w.r.t to the target y.
    :param X: pd.DataFrame, features
    :param y: pd.DataFrame, labels (0/1)
    :return:
        - X_train: pd.DataFrame, training features
        - X_test: pd.DataFrame, testing features
        - y_train: pd.DataFrame, training labels
        - y_test: pd.DataFrame, testing labels
    """
    n = X.shape[0]
    idx_0, idx_1 = np.where(y.loc[:, "Bound"] == -1)[0], np.where(y.loc[:, "Bound"] == 1)[0]
    n0, n1 = len(idx_0), len(idx_1)
    idx_tr0 = np.random.choice(idx_0, int(p * n0), replace=False)
    idx_tr1 = np.random.choice(idx_1, int(p * n1), replace=False)
    idx_te0 = list(set(idx_0) - set(idx_tr0))
    idx_te1 = list(set(idx_1) - set(idx_tr1))
    idx_tr = np.random.permutation(np.concatenate((idx_tr0, idx_tr1)))
    idx_te = np.random.permutation(np.concatenate((idx_te0, idx_te1)))
    n_tr, n_te = len(idx_tr), len(idx_te)
    X_train, y_train = X.iloc[idx_tr, :], y.iloc[idx_tr, :]
    X_test, y_test = X.iloc[idx_te, :], y.iloc[idx_te, :]
    print('Number of training samples: {} ({:0.2f}%), testing samples: {} ({:0.2f}%)'.
          format(n_tr, n_tr / n * 100, n_te, n_te / n * 100))
    print('Count train : -1 ({:0.4f}%), 1 ({:0.4f}%)'.
          format(np.count_nonzero(y_train.loc[:, "Bound"] == -1) / n_tr * 100,
                 np.count_nonzero(y_train.loc[:, "Bound"] == 1) / n_tr * 100))
    print('Count test : -1 ({:0.4f}%), 1 ({:0.4f}%)\n'.
          format(np.count_nonzero(y_test.loc[:, "Bound"] == -1) / n_te * 100,
                 np.count_nonzero(y_test.loc[:, "Bound"] == 1) / n_te * 100))
    return X_train, y_train, X_test, y_test


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
    for each of the 3 types of TF and concatenate trains and tests.
    Training and testing data sets are balanced w.r.t to the target y.
    :return:
        - X_train: pd.DataFrame, training features
        - X_test: pd.DataFrame, testing features
        - y_train: pd.DataFrame, training labels
        - y_test: pd.DataFrame, testing labels
    """
    for k in range(3):
        print('k = ' + str(k))
        X, y = get_train(k)
        if k == 0:
            X_train, y_train, X_test, y_test = train_test_split_k(X, y, 0.75)
        else:
            X_tr, y_tr, X_te, y_te = train_test_split_k(X, y, 0.75)
            X_train = pd.concat((X_train, X_tr), axis=0)
            X_test = pd.concat((X_test, X_te), axis=0)
            y_train = pd.concat((y_train, y_tr), axis=0)
            y_test = pd.concat((y_test, y_te), axis=0)
        X_train, X_test, y_train, y_test = resetIndex([X_train, X_test, y_train, y_test])
    print('Final shape: train {}, test {}'.format(X_train.shape, X_test.shape))
    return X_train, y_train, X_test, y_test


def get_training_datas(method, param, replace=False):
    """
    Construct training and testing data, and kernels.
    :param: d: int, maximal degree for Weighted Degree Kernel
    :param: method: string, method used for computing kernels
    :param: replace: Boolean, whether or not replace the existing files in the repo
    :return:
        - X_train: pd.DataFrame, training features
        - X_test: pd.DataFrame, testing features
        - y_train: pd.DataFrame, training labels
        - y_test: pd.DataFrame, testing labels
        - K_train: np.array, training kernel
        - K_test: np.array, testing kernel
    """
    warnings.filterwarnings('ignore')
    file = 'training_data_'+method+'.pkl'
    if trainInRepo(file) and not replace:
        X_train, y_train, X_test, y_test, K = pkl.load(open(os.path.join('./Data', file), 'rb'))
    else:
        X_train, y_train, X_test, y_test = train_test_split()
        X_train, X_test = X_train.sample(frac=1), X_test.sample(frac=1)
        idx_train, idx_test = X_train.index, X_test.index
        y_train, y_test = y_train.iloc[idx_train, :], y_test.iloc[idx_test, :]
        X = resetIndex([pd.concat((X_train, X_test))])[0]
        K = km.select_method(X, method, param)
        file = 'training_data_'+method+'.pkl'
        pkl.dump([X_train, y_train, X_test, y_test, K], open(os.path.join('./Data', file), 'wb'))
    warnings.simplefilter('always')
    X_train, X_test, y_train, y_test = resetIndex([X_train, X_test, y_train, y_test])
    X_test.index = X_train.shape[0] + np.arange(X_test.shape[0])
    y_test.index = y_train.shape[0] + np.arange(y_test.shape[0])
    return X_train, y_train, X_test, y_test, K


def select_k(k, X_train, y_train, X_test, y_test, K):
    """
    Restrict training and testing data to 1 data set of interest, defined by k (TF type)
    :param k: int, which data set to restrict on
    :param X_train: pd.DataFrame, training features
    :param y_train: pd.DataFrame, training labels
    :param X_test: pd.DataFrame, testing features
    :param y_test: pd.DataFrame, testing labels
    :param K_train: np.array, training kernel
    :return: pd.DataFrames and kernels
    """
    idx_train = np.where(X_train.loc[:, 'k'] == k)[0]
    idx_test = np.where(X_test.loc[:, 'k'] == k)[0]
    idx = np.concatenate((idx_train, idx_test))
    X_train_ = X_train.iloc[idx_train]
    y_train_ = y_train.iloc[idx_train]
    y_test_ = y_test.iloc[idx_test]
    X_test_ = X_test.iloc[idx_test]
    K_ = K[idx][:, idx]
    return X_train_, y_train_, X_test_, y_test_, K_

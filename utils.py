import pandas as pd
import numpy as np
import os
import warnings
import kernels as km
import pickle as pkl
import datetime
import SVM
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
    X_val, y_val = X.iloc[idx_te, :], y.iloc[idx_te, :]
    print('Number of training samples: {} ({:0.2f}%), testing samples: {} ({:0.2f}%)'.
          format(n_tr, n_tr / n * 100, n_te, n_te / n * 100))
    print('Count train : -1 ({:0.4f}%), 1 ({:0.4f}%)'.
          format(np.count_nonzero(y_train.loc[:, "Bound"] == -1) / n_tr * 100,
                 np.count_nonzero(y_train.loc[:, "Bound"] == 1) / n_tr * 100))
    print('Count val : -1 ({:0.4f}%), 1 ({:0.4f}%)\n'.
          format(np.count_nonzero(y_val.loc[:, "Bound"] == -1) / n_te * 100,
                 np.count_nonzero(y_val.loc[:, "Bound"] == 1) / n_te * 100))
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
        print('k = ' + str(k))
        X, y = get_train(k)
        if k == 0:
            X_train, y_train, X_val, y_val = train_test_split_k(X, y, 0.75)
            X_test = get_test(k)
        else:
            X_tr, y_tr, X_v, y_te = train_test_split_k(X, y, 0.75)
            X_train = pd.concat((X_train, X_tr), axis=0)
            X_val = pd.concat((X_val, X_v), axis=0)
            y_train = pd.concat((y_train, y_tr), axis=0)
            y_val = pd.concat((y_val, y_te), axis=0)
            X_test = pd.concat((X_test, get_test(k)), axis=0)
        X_train, X_val, y_train, y_val, X_test = resetIndex([X_train, X_val, y_train, y_val, X_test])
    print('Final shape: train {}, val {}, test {}'.format(X_train.shape, X_val.shape, X_test.shape))
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
        X_train, X_val = X_train.sample(frac=1), X_val.sample(frac=1)
        idx_train, idx_val = X_train.index, X_val.index
        y_train, y_val = y_train.iloc[idx_train, :], y_val.iloc[idx_val, :]
        X_test.loc[:, 'Id'] = -(X_test.loc[:, 'Id'] + 1)
        X = pd.concat((X_train, X_val, X_test), axis=0)
        ID = X.loc[:, 'Id']
        return X_train, y_train, X_val, y_val, X_test, ID
    else:
        if trainInRepo(file) and not replace:
            X_train, y_train, X_val, y_val, X_test, K, ID = pkl.load(open(os.path.join('./Data', file), 'rb'))
        else:
            X_train, y_train, X_val, y_val, X_test = train_test_split()
            X_train, X_val = X_train.sample(frac=1), X_val.sample(frac=1)
            idx_train, idx_val = X_train.index, X_val.index
            y_train, y_val = y_train.iloc[idx_train, :], y_val.iloc[idx_val, :]
            X_test.loc[:, 'Id'] = -(X_test.loc[:, 'Id']+1)
            X = pd.concat((X_train, X_val, X_test), axis=0)
            ID = np.array(X.loc[:, 'Id'])
            K = km.select_method(X, method)
            file = 'training_data_'+method+'.pkl'
            pkl.dump([X_train, y_train, X_val, y_val, X_test, K, ID], open(os.path.join('./Data', file), 'wb'))
        warnings.simplefilter('always')
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
    id_train = X_train.iloc[idx_train, 0]
    id_val = X_val.iloc[idx_val, 0]
    id_test = X_test.iloc[idx_test, 0]
    id_k = np.concatenate((id_train, id_val, id_test))
    idx = np.where(np.in1d(ID, id_k))[0]
    X_train_ = X_train.iloc[idx_train]
    y_train_ = y_train.iloc[idx_train]
    y_val_ = y_val.iloc[idx_val]
    X_val_ = X_val.iloc[idx_val]
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
        if k==0:
            y_test = pd.DataFrame({'Id': X_test.Id, 'Bound': pred_test})
        else:
            y_test = pd.concat((y_test, pd.DataFrame({'Id': X_test.Id, 'Bound': pred_test})))
    y_test.Id = -y_test.Id - 1
    y_test.Bound = y_test.Bound.replace(-1, 0)
    t = datetime.datetime.now().time()
    y_test.to_csv('y_test_' + str(t) + '.csv', index=False)
    return y_test


def sort_accuracies(algo='C_SVM', k=1):
    k_ = 'k'+str(k)
    val_scores = {}
    C_opts = {}
    warnings.filterwarnings('ignore')
    for file in os.listdir('./Data'):
        if algo in file and k_ in file:
            pred = pkl.load(open(os.path.join('./Data', file), 'rb'))
            file = file.split(k_)[0][5:-1]
            val_scores[file] = np.max(pred[3])
            C_opts[file] = pred[4]
    sorted_val = sorted(val_scores.items(), key=operator.itemgetter(1), reverse=True)
    sorted_C = {}
    for i in range(len(sorted_val)):
        key = sorted_val[i][0]
        sorted_C[key] = C_opts[key]
    u = sorted_val, sorted_C
    p = pd.DataFrame({'Kernel Method': [u[0][i][0] for i in range(len(u[0]))],
                      'Val accuracy': [u[0][i][1] for i in range(len(u[0]))],
                      'Constant C': [np.round(i, 4) for i in u[1].values()]})
    return p


Cs = np.sort([i*10**j for (i,j) in product(range(1,10), range(-3,1))])


def run_expe(methods, k=3, maxiter=500, kfolds=5, Cs_1=Cs, Cs_2=Cs, Cs_3=Cs):
    for m in methods:
        X_train, y_train, X_val, y_val, X_test, K, ID = get_training_datas(method=m, all=True, replace=False)
        for k_ in range(1, k+1):
            exec("X_train_"+str(k_)+", y_train_"+str(k_)+", X_val_"+str(k_)+", y_val_"+str(k_)+", X_test_"+str(k_)+", K_"+str(k_)+", id_"+str(k_)+" = select_k(k_, X_train, y_train, X_val, y_val, X_test, K, ID)")
            exec("data_"+str(k_)+" = [X_train_"+str(k_)+", y_train_"+str(k_)+", X_val_"+str(k_)+", y_val_"+str(k_)+"]")
            pickleName = 'cv_C_SVM_'+m+'_k'+str(k_)+'_max_iter'+str(maxiter)+'_solver_BFGS_full.pkl'
            exec("C_opt_"+str(k_)+", _, _, _, _ = SVM.cv(Cs=Cs_"+str(k_)+", data=data_"+str(k_)+", kfolds=kfolds, pickleName=pickleName, K=K, ID=ID, maxiter=maxiter, method='BFGS')")


import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import os
import pickle as pkl
import warnings
from scipy.optimize import fmin_l_bfgs_b
warnings.filterwarnings('ignore')


class C_SVM():
    """
    Implementation of C-SVM algorithm
    """
    def __init__(self, K, ID, C=10, eps=1e-5, tol=1e-4, print_callbacks=True):
        """
        :param K: np.array, kernel (computed on train+val+test data sets)
        :param ID: np.array, Ids (for ordering)
        :param C: float, regularization constant
        :param eps: float, threshold determining whether alpha is a support vector or not
        :param tol: float, stopping criteria for gradient descent
        :param print_callbacks: Bool, print evolution of gradient descent (suggested)
        """
        self.K = K
        self.ID = ID
        self.C = C
        self.tol = tol
        self.eps = eps
        self.print_callbacks = print_callbacks
        self.Nfeval = 1

    def loss(self, a):
        """
        :param a: np.array, alphas
        :return: float, loss function
        """
        return -(2 * np.dot(a, self.y_fit) - np.dot(a.T, np.dot(self.K_fit, a)))

    def jac(self, a):
        """
        :param a: np.array, alphas
        :return: np.array, loss Jacobian
        """
        return -(2 * self.y_fit - 2*np.dot(self.K_fit, a))

    def callbackF(self, Xi, Yi=0):
        """
        Print useful information about gradient descent evolution. This function aims at selecting the iteration
        for which the accuracy on the validation set was the best (hence the best vectors alphas).
        :param Xi: np.array, values returned by scipy.minimize at each iteration
        :return: None, update print
        """
        if self.print_callbacks:
            if self.Nfeval == 1:
                self.L = self.loss(Xi)
                print('Iteration {0:2.0f} : loss={1:8.4f}'.format(self.Nfeval, self.L))
            else:
                l_next = self.loss(Xi)
                print('Iteration {0:2.0f} : loss={1:8.4f}, tol={2:8.4f}'
                      .format(self.Nfeval, l_next, abs(self.L - l_next)))
                self.L = l_next
            self.Nfeval += 1
        else:
            self.Nfeval += 1

    def fit(self, X, y):
        """
        Train C-SVM on X and y. X_pred and y_pred are used for prediction.
        :param X: pd.DataFrame, training features
        :param y: pd.DataFrame, training labels
        :return:
        """
        self.Id_fit = np.array(X.loc[:, 'Id'])
        self.idx_fit = np.where(np.in1d(self.ID, self.Id_fit))[0]
        self.K_fit = self.K[self.idx_fit][:, self.idx_fit]
        self.y_fit, self.X_fit, = np.array(y.loc[:, 'Bound']), X
        n = self.K_fit.shape[0]
        # initialization
        a0 = np.zeros(n)
        # Gradient descent
        bounds_down = [-self.C if self.y_fit[i] <= 0 else 0 for i in range(n)]
        bounds_up = [+self.C if self.y_fit[i] >= 0 else 0 for i in range(n)]
        bounds = [[bounds_down[i], bounds_up[i]] for i in range(n)]
        res = fmin_l_bfgs_b(self.loss, a0, fprime=self.jac, bounds=bounds, callback=self.callbackF)
        # Align support vectors index with index from fit set
        self.idx_sv = np.where(np.abs(res[0]) > self.eps)
        self.sv = res[0][self.idx_sv]
        self.idx_sv = self.idx_fit[self.idx_sv]

    def predict(self, X):
        """
        Make predictions for features in X
        :param X: pd.DataFrame, features
        :return: np.array, predictions (-1/1)
        """
        # Align prediction IDs with index in kernel K
        self.Id_pred = np.array(X.loc[:, 'Id'])
        self.idx_pred = np.where(np.in1d(self.ID, self.Id_pred))[0]
        self.idx_tot = np.unique(np.concatenate((self.idx_fit, self.idx_pred)))
        pred = []
        for i in self.idx_pred:
            pred.append(np.sign(np.dot(self.sv, self.K[self.idx_sv, i].squeeze())))
        return np.array(pred)

    def score(self, pred, y):
        """
        Compute accuracy of predictions according to y
        :param pred: np.array, predictions (-1/1)
        :param y: np.array or pd.DataFrame, true labels
        :return: float, percentage of correct predictions
        """
        if not isinstance(y, np.ndarray):
            label = np.array(y.loc[:, 'Bound'])
        else:
            label = y
        assert 0 not in np.unique(label), "Labels must be -1 or 1, not 0 or 1"
        return np.mean(pred == label)


def cv(Cs, data, kfolds=5, pickleName='cv_C_SVM', **kwargs):
    """
    Cross-validation implementation for C_SVM
    :param Cs: list or np.array, list of constant C to loop on for gridsearch
    :param data: list, X_train + y_train + X_val + y_val
    :param kfolds: int, number of folds used for CV
    :param pickleName: string
    :param kwargs: arguments used for C_SVM (maxiter, tol, etc...)
    :return: list: - C_opt: float, optimal constant (best average score)
                   - scores_tr: np.array, scores on train set for each constant C (and each fold)
                   - scores_te: np.array, scores on val set for each constant C (and each fold)
                   - mean_scores_tr: np.array, average scores on train set for each constant C
                   - mean_scores_te: np.array, average scores on val set for each constant C
    """
    scores_tr = np.zeros((kfolds, len(Cs)))
    scores_te = np.zeros((kfolds, len(Cs)))
    X_tr, y_tr, X_te, y_te = data
    X_train_ = pd.concat((X_tr, X_te)).reset_index(drop=True)
    y_train_ = pd.concat((y_tr, y_te)).reset_index(drop=True)
    n = X_train_.shape[0]
    p = int(n // kfolds)
    for k in tqdm(range(kfolds)):
        print('Fold {}'.format(k+1))
        q = p * (k + 1) + n % kfolds if k == kfolds - 1 else p * (k + 1)
        idx_val = np.arange(p * k, q)
        idx_train = np.setdiff1d(np.arange(n), idx_val)
        X_train = X_train_.iloc[idx_train, :]
        y_train = y_train_.iloc[idx_train, :]
        X_val = X_train_.iloc[idx_val, :]
        y_val = y_train_.iloc[idx_val, :]
        s_tr, s_te = [], []
        for C in Cs:
            svm = C_SVM(C=C, print_callbacks=False, **kwargs)
            svm.fit(X_train, y_train, X_val, y_val)
            pred_tr = svm.predict(X_train)
            score_tr = svm.score(pred_tr, y_train)
            pred_te = svm.predict(X_val)
            score_te = svm.score(pred_te, y_val)
            s_tr.append(score_tr)
            s_te.append(score_te)
            print('C={}, best accuracy on train ({:0.4f}) and val ({:0.4f})'.format(C, score_tr, score_te))
        scores_tr[k], scores_te[k] = s_tr, s_te
    mean_scores_tr, mean_scores_te = np.mean(scores_tr, axis=0), np.mean(scores_te, axis=0)
    C_opt = Cs[np.argmax(mean_scores_te)]
    print('Best constant C: {}, accuracy on val {:0.4f}'.format(C_opt, np.max(mean_scores_te)))
    pkl.dump([scores_tr, scores_te, mean_scores_tr, mean_scores_te, C_opt],
             open(os.path.join('./Data/CrossVals/', pickleName), 'wb'))
    return C_opt, scores_tr, scores_te, mean_scores_tr, mean_scores_te

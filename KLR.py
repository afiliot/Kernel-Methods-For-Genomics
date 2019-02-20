import numpy as np
import copy
from tqdm import tqdm
import os
import pickle as pkl
import pandas as pd

class KLR():
    def __init__(self, K, ID,eps=1e-5, lbda=0.1, tol=1e-5, maxiter=50, print_callbacks=True):
        """
        kernel specifies the type of kernel to use
        lamb specifies the regularization parameter
        """
        self.K = K
        self.ID = ID
        self.eps = eps
        self.lbda = lbda
        self.tol = tol
        self.maxiter = maxiter
        self.print_callbacks = print_callbacks
        self.Nfeval = 1

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def IRLS(self, K, y, alpha):
        """
        itertive step to update alpha when training the classifier
        """
        m = np.dot(K, alpha)
        W = self.sigmoid(m) * self.sigmoid(-m)
        z = m + y/self.sigmoid(-y*m)
        return W, z

    def WKRR(self, K, W, z):
        W_s = np.diag(np.sqrt(W))
        A = np.dot(np.dot(W_s, K), W_s) + self.n * self.lbda * np.eye(self.n)
        A = np.dot(np.dot(W_s, np.linalg.inv(A)), W_s)
        return np.dot(A, z)

    def fit(self, X, y):
        self.Id_fit = np.array(X.loc[:, 'Id'])
        self.idx_fit = np.array([np.where(self.ID == self.Id_fit[i])[0] for i in range(len(self.Id_fit))]).squeeze()
        self.K_fit = self.K[self.idx_fit][:, self.idx_fit]
        self.y_fit, self.X_fit, = np.array(y.loc[:, 'Bound']), X
        self.n = self.K_fit.shape[0]
        alpha_prev = np.zeros(self.n)
        diff = np.inf
        for _ in range(self.maxiter):
            if diff > self.tol:
                W, z = self.IRLS(self.K_fit, self.y_fit, alpha_prev)
                alpha = self.WKRR(self.K_fit, W, z)
                diff = np.linalg.norm(alpha-alpha_prev, ord=2)
                alpha_prev = copy.copy(alpha)
        self.a = alpha_prev
        # Align support vectors index with index from fit set
        self.idx_sv = np.where(np.abs(self.a) > self.eps)
        self.y_fit = self.y_fit[self.idx_sv]
        self.a = self.a[self.idx_sv]
        self.idx_sv = self.idx_fit[self.idx_sv]
        # Intercept
        self.y_hat = np.array([np.dot(self.a, self.K[self.idx_sv, i]).squeeze() for i in self.idx_sv])
        self.b = np.mean(self.y_fit - self.y_hat)

    def predict(self, X):
        """
        Make predictions for features in X
        :param X: pd.DataFrame, features
        :return: np.array, predictions (-1/1)
        """
        # Align prediction IDs with index in kernel K
        self.Id_pred = np.array(X.loc[:, 'Id'])
        self.idx_pred = np.array([np.where(self.ID == self.Id_pred[i])[0] for i in range(len(self.Id_pred))]).squeeze()
        pred = []
        for i in self.idx_pred:
            pred.append(np.sign(np.dot(self.a, self.K[self.idx_sv, i].squeeze()) + self.b))
        return np.array(pred)

    def score(self, pred, y):
        """
        Compute accuracy of predictions according to y
        :param pred: np.array, predictions (-1/1)
        :param y: np.array or pd.DataFrame, true labels
        :return: float, percentage of correct predictions
        """
        label = np.array(y.loc[:, 'Bound']) if not isinstance(y, np.ndarray) else y
        assert 0 not in np.unique(label), "Labels must be -1 or 1, not 0 or 1"
        return np.mean(pred == label)


def cv(Ls, data, kfolds=5, pickleName='cv_KLR_', **kwargs):
    """
    Cross-validation implementation for C_SVM
    :param Cs: list or np.array, list of constant C to loop on for gridsearch
    :param data: list, X_train + y_train + X_val + y_val
    :param kfolds: int, number of folds used for CV
    :param pickleName: string
    :param kwargs: arguments used for C_SVM (tol, etc...)
    :return: list: - C_opt: float, optimal constant (best average score)
                   - scores_tr: np.array, scores on train set for each constant C (and each fold)
                   - scores_te: np.array, scores on val set for each constant C (and each fold)
                   - mean_scores_tr: np.array, average scores on train set for each constant C
                   - mean_scores_te: np.array, average scores on val set for each constant C
    """
    scores_tr = np.zeros((kfolds, len(Ls)))
    scores_te = np.zeros((kfolds, len(Ls)))
    X_tr, y_tr, X_te, y_te = data
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
        for l in Ls:
            svm = KLR(lbda=l, **kwargs)
            svm.fit(X_train, y_train)
            pred_tr = svm.predict(X_train)
            score_tr = svm.score(pred_tr, y_train)
            pred_te = svm.predict(X_val)
            score_te = svm.score(pred_te, y_val)
            s_tr.append(score_tr)
            s_te.append(score_te)
            print('lambda={}, train_acc={:0.4f}, val_acc={:0.4f}'.format(l, score_tr, score_te))
        scores_tr[k], scores_te[k] = s_tr, s_te
    mean_scores_tr, mean_scores_te = np.mean(scores_tr, axis=0), np.mean(scores_te, axis=0)
    l_opt = Ls[np.argmax(mean_scores_te)]
    print('Best constant lambda: {}, accuracy on val {:0.4f}'.format(l_opt, np.max(mean_scores_te)))
    pkl.dump([scores_tr, scores_te, mean_scores_tr, mean_scores_te, l_opt],
             open(os.path.join('./Data/CrossVals/', pickleName), 'wb'))
    return l_opt, scores_tr, scores_te, mean_scores_tr, mean_scores_te
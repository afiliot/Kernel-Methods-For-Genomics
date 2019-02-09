import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import os
import pickle as pkl
from scipy.optimize import minimize


class C_SVM():
    def __init__(self, K, ID, C=10, maxiter=1000, tol=1e-4, eps=1e-4, method='SLSQP', print_callbacks=True):
        self.K = K
        self.ID = ID
        self.C = C
        self.maxiter = maxiter
        self.tol = tol
        self.eps = eps
        self.method = method
        self.print_callbacks = print_callbacks
        self.Nfeval = 1

    def loss(self, a):
        return -(2 * np.dot(a, self.y_fit) - np.dot(a.T, np.dot(self.K_fit, a)))

    def jac(self, a):
        return -(2 * self.y_fit - 2 * (np.dot(self.K_fit, a)))

    def hess(self, a):
        return -(-2 * self.K_fit)

    def callbackF(self, Xi):
        """
        :param Xi: np.array, values returned by scipy.minimize at each iteration
        :return: None, update print
        """
        self.idx_sv = np.where(Xi > self.eps)
        self.sv = Xi[self.idx_sv]
        score = self.score(self.predict(self.X_fit), self.y_fit)
        if self.Nfeval == 1:
            self.L = self.loss(Xi)
            print('Iteration {0:2.0f} : S(y)={1:}'.format(self.Nfeval, self.L))
        else:
            l_next = self.loss(Xi)
            print('Iteration {0:2.0f} : S(y)={1:}, tol={2:}, acc={3:0.4f}'.format(self.Nfeval, l_next, abs(self.L - l_next), score))
            self.L = l_next
        self.Nfeval += 1

    def fit(self, X, y):
        self.Id_fit = np.array(X.loc[:, 'Id'])
        self.idx_fit = np.where(np.in1d(self.ID, self.Id_fit))[0]
        self.K_fit = self.K[self.idx_fit][:, self.idx_fit]
        self.y_fit = np.array(y.loc[:, 'Bound'])
        self.X_fit = X
        n = self.K_fit.shape[0]
        a0 = np.random.randn(n)
        constraints = []
        for i in range(n):
            constraints.append({'type': 'ineq',
                                'fun': lambda a: self.C - a[i] * self.y_fit[i],
                                'jac': lambda a: -self.y_fit})
            constraints.append({'type': 'ineq',
                                'fun': lambda a: a[i] * self.y_fit[i],
                                'jac': lambda a: self.y_fit})
        constraints = tuple(constraints)
        if self.method == 'SLSQP':
            if self.print_callbacks:
                print('Starting gradient descent:\n')
                res = minimize(self.loss, a0, jac=self.jac,
                               constraints=constraints, method='SLSQP',
                               tol=self.tol, callback=self.callbackF,
                               options={'maxiter': self.maxiter})
            else:
                res = minimize(self.loss, a0, jac=self.jac,
                               constraints=constraints, method='SLSQP',
                               tol=self.tol, options={'maxiter': self.maxiter})
        self.idx_sv = np.where(res.x > self.eps)
        self.sv = res.x[self.idx_sv]
        return self.sv

    def predict(self, X):
        self.Id_pred = np.array(X.loc[:, 'Id'])
        self.idx_pred = np.where(np.in1d(self.ID, self.Id_pred))[0]
        self.idx_tot = np.unique(np.concatenate((self.idx_fit, self.idx_pred)))
        pred = []
        for i in self.idx_pred:
            pred.append(np.sign(np.dot(self.sv, self.K[self.idx_sv, i].squeeze())))
        return np.array(pred)

    def score(self, pred, y):
        if not isinstance(y, np.ndarray):
            label = np.array(y.loc[:, 'Bound'])
        else:
            label = y
        assert 0 not in np.unique(label), "Labels must be -1 or 1, not 0 or 1"
        return np.mean(pred == label)

    def cv(self, Cs, data, kfolds=5, pickleName='cv_C_SVM'):
        scores_tr = np.zeros((kfolds, len(Cs)))
        scores_te = np.zeros((kfolds, len(Cs)))
        X_tr, y_tr, X_te, y_te = data
        X_train_ = pd.concat((X_tr, X_te)).reset_index(drop=True)
        y_train_ = pd.concat((y_tr, y_te)).reset_index(drop=True)
        n = X_train_.shape[0]
        p = int(n // kfolds)
        for k in tqdm(range(kfolds)):
            print('Fold {}'.format(k))
            q = p * (k + 1) + n % kfolds if k == kfolds - 1 else p * (k + 1)
            idx_val = np.arange(p * k, q)
            idx_train = np.setdiff1d(np.arange(n), idx_val)
            X_train = X_train_.iloc[idx_train, :]
            y_train = y_train_.iloc[idx_train, :]
            X_val = X_train_.iloc[idx_val, :]
            y_val = y_train_.iloc[idx_val, :]
            s_tr, s_te = [], []
            for C in Cs:
                svm = C_SVM(self.K, self.ID, C=C, print_callbacks=False); svm.fit(X_train, y_train)
                pred_tr = svm.predict(X_train)
                score_tr = svm.score(pred_tr, y_train)
                pred_te = svm.predict(X_val)
                score_te = svm.score(pred_te, y_val)
                s_tr.append(score_tr)
                s_te.append(score_te)
                print('C={}, accuracy on train ({:0.4f}) and val ({:0.4f})'.format(C, score_tr, score_te))
            scores_tr[k], scores_te[k] = s_tr, s_te
        mean_scores_tr, mean_scores_te = np.mean(scores_tr, axis=0), np.mean(scores_te, axis=0)
        C_opt = Cs[np.argmax(mean_scores_te)]
        print('Best constant C: {}, accuracy on val {:0.4f}'.format(C_opt, np.max(mean_scores_te)))
        pkl.dump([scores_tr, scores_te, mean_scores_tr, mean_scores_te, C_opt],
                 open(os.path.join('./Data', pickleName+'.pkl'), 'wb'))
        return C_opt, scores_tr, scores_te, mean_scores_tr, mean_scores_te
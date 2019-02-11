import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import os
import pickle as pkl
from scipy.optimize import minimize, LinearConstraint, Bounds
import warnings


class C_SVM():
    def __init__(self, K, ID, C=10, maxiter=1000, eps=1e-5, tol=1e-4, method='BFGS', print_callbacks=True):
        self.K = K
        self.ID = ID
        self.C = C
        self.maxiter = maxiter
        self.tol = tol
        self.eps = eps
        self.method = method
        self.print_callbacks = print_callbacks
        self.Nfeval = 1
        self.val_accuracies = []
        self.idx_svS = []
        self.svS = []

    def loss(self, a):
        return -(2 * np.dot(a, self.y_fit) - np.dot(a.T, np.dot(self.K_fit, a)))

    def jac(self, a):
        return -(2 * self.y_fit - 2 * (np.dot(self.K_fit, a)))

    def hess(self, a):
        return -(-2 * self.K_fit)

    def callbackF(self, Xi, Yi=0):
        """
        :param Xi: np.array, values returned by scipy.minimize at each iteration
        :return: None, update print
        """
        self.idx_sv = np.where(np.abs(Xi) > self.eps)
        self.sv = Xi[self.idx_sv]
        self.idx_sv = self.idx_fit[self.idx_sv]
        train_acc = self.score(self.predict(self.X_fit), self.y_fit)
        if self.X_pred is None:
            val_acc = 0
        else:
            val_acc = self.score(self.predict(self.X_pred), self.y_pred)
            self.val_accuracies.append(val_acc)
        self.svS.append(self.sv)
        self.idx_svS.append(self.idx_sv)
        if self.print_callbacks:
            if self.Nfeval == 1:
                self.L = self.loss(Xi)
                print('Iteration {0:2.0f} : loss={1:8.4f}'.format(self.Nfeval, self.L))
            else:
                l_next = self.loss(Xi)
                print('Iteration {0:2.0f} : loss={1:8.4f}, tol={2:8.4f}, train_acc={3:0.4f}, val_acc={4:0.4f}'
                      .format(self.Nfeval, l_next, abs(self.L - l_next), train_acc, val_acc))
                self.L = l_next
            self.Nfeval += 1
        else:
            self.Nfeval += 1

    def fit(self, X, y, X_pred=None, y_pred=None):
        self.Id_fit = np.array(X.loc[:, 'Id'])
        self.idx_fit = np.where(np.in1d(self.ID, self.Id_fit))[0]
        self.K_fit = self.K[self.idx_fit][:, self.idx_fit]
        self.y_fit, self.X_fit, self.X_pred, self.y_pred = np.array(y.loc[:, 'Bound']), X, X_pred, y_pred
        n = self.K_fit.shape[0]
        a0 = np.zeros(n)
        if self.method == 'trust':
            warnings.filterwarnings('ignore')
            constraints = LinearConstraint(np.diag(self.y_fit), np.zeros(n), self.C * np.ones(n))
            res = minimize(self.loss, a0, jac=self.jac, hess=self.hess,
                           constraints=constraints, method='trust-constr',
                           callback=self.callbackF,
                           tol=self.tol, options={'maxiter': self.maxiter})
        elif self.method == 'SLSQP':
            Y = np.diag(self.y_fit)
            constraints = ({'type': 'ineq', 'fun': lambda x: self.C*np.ones(n) - np.dot(Y, x), 'jac': lambda x: -Y},
                           {'type': 'ineq', 'fun': lambda x: np.dot(Y, x), 'jac': lambda x: Y})
            res = minimize(self.loss, a0, jac=self.jac,
                           constraints=constraints, method='SLSQP',
                           callback=self.callbackF,
                           tol=self.tol, options={'maxiter': self.maxiter})
        elif self.method == "BFGS":
            bounds = Bounds(np.array([-self.C if self.y_fit[i] <= 0 else 0 for i in range(n)]),
                            np.array([+self.C if self.y_fit[i] >= 0 else 0 for i in range(n)]))
            res = minimize(self.loss, a0, jac=self.jac,
                           bounds=bounds, method='L-BFGS-B',
                           callback=self.callbackF,
                           tol=self.tol, options={'maxiter': self.maxiter})
        elif self.method == "Newton":
            bounds = Bounds(np.array([-self.C if self.y_fit[i] <= 0 else 0 for i in range(n)]),
                            np.array([+self.C if self.y_fit[i] >= 0 else 0 for i in range(n)]))
            res = minimize(self.loss, a0, jac=self.jac,
                           bounds=bounds, method='TNC',
                           callback=self.callbackF,
                           tol=self.tol, options={'maxiter': self.maxiter})
        # Select the alphas which led to the best accuracy on validation set
        best_val_idx = np.argmax(np.array(self.val_accuracies))
        self.sv = self.svS[best_val_idx]
        self.idx_sv = self.idx_svS[best_val_idx]
        warnings.filterwarnings('always')
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


def cv(Cs, data, kfolds=5, pickleName='cv_C_SVM', **kwargs):
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
             open(os.path.join('./Data', pickleName), 'wb'))
    return C_opt, scores_tr, scores_te, mean_scores_tr, mean_scores_te

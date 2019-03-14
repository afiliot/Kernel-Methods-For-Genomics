import numpy as np
from cvxopt import matrix, spmatrix, solvers
import utils
from kernels import normalize_K
from itertools import product
import pandas as pd
import pickle as pkl
import os
from tqdm import tqdm as tqdm
solvers.options['show_progress'] = False

class NLCK():
    """
    Implementation of NLCK algorithm.
    Reference : "Learning Non-Linear Combinations of Kernels", Cortes et al. (2009)
    Notations from the original article has been fully reused.
    NLCK class returns the optimal weights of the non-linear combination of kernels
    """
    def __init__(self, X, y, ID, kernels, C=1e-5, eps=1e-8, degree=2):
        """
        :param X: pd.DataFrame, training features
        :param y: pd.DataFrame, training labels
        :param ID: np.array, Ids (for ordering)
        :param kernels: list of kernels
        :param C: float, float, regularization constant for C-SVM
        :param eps: float, threshold determining whether alpha is a support vector or not
        :param degree: int, order of polynomial combination
        """
        self.X = X
        self.y = y.loc[:, 'Bound']
        self.n = y.shape[0]
        self.ID = ID
        self.kernels = self.normalize_kernels(kernels)
        self.Id_X = np.array(X.loc[:, 'Id'])
        self.idx = np.array([np.where(self.ID == self.Id_X[i])[0] for i in range(len(self.Id_X))]).squeeze()
        self.kernels_fit = [K[self.idx][:, self.idx] for K in self.kernels]
        self.p = len(self.kernels_fit)
        self.C = C
        self.lbda = 1/(2*self.C*self.n)
        self.eps = eps
        self.degree = degree

    def normalize_kernels(self, kernels):
        new_kernels = []
        for k, K in enumerate(kernels):
            print('Normalizing kernel {}...'.format(k + 1))
            new_kernels.append(normalize_K(K))
        return new_kernels

    def svm_step(self, u):
        r, o, z = np.arange(self.n), np.ones(self.n), np.zeros(self.n)
        K = np.sum((self.kernels_fit * u[:, None, None]), axis=0) ** self.degree
        P = matrix(K.astype(float), tc='d')
        q = matrix(-self.y, tc='d')
        G = spmatrix(np.r_[self.y, -self.y], np.r_[r, r + self.n], np.r_[r, r], tc='d')
        h = matrix(np.r_[o * self.C, z], tc='d')
        sol = solvers.qp(P, q, G, h)
        a = np.ravel(sol['x'])
        return a

    def grad(self, u, alpha):
        K_t = np.sum(self.kernels_fit * u[:, None, None], axis=0) ** (self.degree - 1)
        grad = np.zeros(self.p)
        for m in range(self.p):
            grad[m] = alpha.T.dot((K_t * self.kernels_fit[m])).dot(alpha)
        return - self.degree * grad

    def normalize(self, u, u0, fnorm):
        u_s = (u - u0)
        u_s_norm = u_s / np.sqrt(np.sum(u_s**2))
        u_s = u_s_norm * fnorm
        return u_s + u0

    def fit(self, u0=0, fnorm=10, n_iter=20, eta=1):
        u = np.ones(self.p)
        u = self.normalize(u, u0, fnorm)
        u = np.array([0 if u[i] < 0 else u[i] for i in range(self.p)])
        score_prev = np.inf
        for k in range(n_iter):
            print('Iteration {}, u={}, score={:0.5f}'.format(k, u, score_prev))
            alpha = self.svm_step(u)
            g = self.grad(u, alpha)
            u_next = self.normalize(u - eta * g, u0, fnorm)
            u_next = np.array([0 if u_next[i] < 0 else u_next[i] for i in range(self.p)])
            score = np.linalg.norm(u_next - u, np.inf)
            if score > score_prev:
                eta *= 0.8
            if score < self.eps:
                return u_next
            u = u_next
            score_prev = score.copy()
        return u_next

    def get_K(self, u0=0, fnorm=1, n_iter=20, eta=1):
        u_star = self.fit(u0, fnorm, n_iter, eta)
        print('Alignment vector : ', u_star)
        Km = np.sum((self.kernels * u_star[:, None, None]), axis=0) ** self.degree
        print('Normalizing final kernel...')
        Km = normalize_K(Km)
        return Km


def cross_validation(k, methods, Cs_NLK, Cs_SVM, degrees, lambdas):
    """
    Apply cross-validation on NLCK algorithm. A first cross-validation is done on the values of C, d and lambda
    in NLCK, in order to find the optimal non-linear combination of kernels along with C, d, lambda. Then,
    for each triplet (and hence the corresponding weights vector), cross validation is done on the regularization
    constant of C_SVM, C.
    :param k: int, which dataset to use (k=1, 2 or 3)
    :param methods: list of string, kernel methods
    :param Cs_NLK: np.array, regularization constants in NLCK algorithm
    :param Cs_SVM: np.array, regularization constants in C_SVM algorithm
    :param degrees: np.array, degrees to explore (usually np.range(1, 5))
    :param lambdas: np.array, lambdas (corresponding to parameter 'fnorm' in NLCK) to explore
    :return: pd.DataFrame with the following columns:
            - 'methods': kernels method used
            - 'C_NLCK': regularization constants in NLCK algorithm
            - 'd': degree in NLCK algorithm
            - 'lambda': normalization parameter in NLCK algorithm
            - 'Best C CSVM': best regularization constant in CSVM after cross validation
            - 'val acc': accuracy obtained on validation set
    """
    # Load data
    data, data1, data2, data3, kernels, ID = utils.get_all_data(methods)
    data_k = [data1, data2, data3]
    # Initialize results DataFrame
    p = len(kernels)
    n_param = len(Cs_NLK) * len(degrees) * len(lambdas)
    init = np.zeros(n_param)
    results = pd.DataFrame({'methods': [methods] * len(init), 'C NLCK': init, 'd': init, 'lambda': init, 'Best C CSVM': init, 'val acc': init})
    # Reformat
    X_train, y_train, X_val, y_val, X_test, kernels, ID = utils.reformat_data(data_k[k-1], kernels, ID)
    # Start cross validation on triplet (C, d, lambda)
    for i, param in tqdm(enumerate(product(Cs_NLK, degrees, lambdas)), total=n_param):
        C, d, lbda = param
        print('NLCK C={}, degree={}, lambda={}'.format(C, d, lbda))
        # Compute kernel
        Km = NLCK(X_train, y_train, ID, kernels, C=C, eps=1e-9, degree=d).get_K(fnorm=lbda)
        # Cross validation on constant C of C-SVM
        C_opt, scores_tr, scores_te, mean_scores_tr, mean_scores_te = \
            utils.cross_validation(Ps=Cs_SVM,
                                   data=[X_train, y_train, X_val, y_val, X_test],
                                   algo='CSVM',
                                   kfolds=3,
                                   K=Km,
                                   ID=ID,
                                   pickleName='cv_C_SVM_NLCK_C{}_d{}_l{}_p{}_k{}.pkl'.format(C, d, lbda, p, k))
        # Save results
        results.iloc[i, 1:6] = C, d, lbda, C_opt, np.max(mean_scores_te)
        pkl.dump(results, open(os.path.join('./Data/CrossVals/', 'cv_C_SVM_NLCK_p{}_k{}.pkl'.format(p, k)), 'wb'))
    return results
import numpy as np
from cvxopt import matrix, spmatrix, solvers
import utils
solvers.options['show_progress'] = False
import kernels
from kernels import normalize_K

class NLCK():

    def __init__(self, X, y, ID, kernels, lbda=0.01, eps=1e-5, degree=4):
        self.X = X
        self.y = y.loc[:, 'Bound']
        self.n = y.shape[0]
        self.ID = ID
        self.kernels = kernels
        self.idx = np.where(np.in1d(self.ID, np.array(self.X.loc[:, 'Id'])))[0]
        self.kernels_fit = [K[self.idx][:, self.idx] for K in self.kernels]
        self.kernels_fit = [normalize_K(K) for K in self.kernels_fit]
        self.p = len(self.kernels_fit)
        self.lbda = lbda
        self.eps = eps
        self.degree = degree
    
    def svm_step(self, u):
        K = np.sum((self.kernels_fit * u[:, None, None]), axis=0) ** self.degree
        return np.dot(np.linalg.inv(K + self.lbda * np.ones(self.n)), self.y)

    def grad(self, u, alpha):
        K_t = np.sum(self.kernels_fit * u[:, None, None], axis=0) ** (self.degree - 1)
        grad = np.zeros(self.p)
        for m in range(self.p):
            grad[m] = alpha.T.dot((K_t * self.kernels_fit[m])).dot(alpha)
        return - self.degree * grad

    def normalize(self, u, u0, fnorm):
        u_s = (u - u0)
        u_s = fnorm * np.abs(u_s) / np.linalg.norm(u_s, ord=2)
        return u_s + u0

    def fit(self, u0=0, fnorm=1, n_iter=20, eta=1):
        u = self.normalize(np.random.randn(self.p), u0, fnorm)
        score_prev = np.inf
        k = 0
        while score_prev > self.eps and k < n_iter:
            k += 1
            print('Iteration {}, u={}, score={:0.5f}'.format(k, u, score_prev))
            alpha = self.svm_step(u)
            g = self.grad(u, alpha)
            u_next = u - eta * g
            u_next = self.normalize(u_next, u0, fnorm)
            score = np.linalg.norm(u_next - u, ord=2)
            if score > score_prev:
                eta *= 0.8
            u = u_next
            score_prev = score
        u = u_next
        return u

    def get_aligned_kernel(self, u0=0, fnorm=1, n_iter=20, eta=1):
        u_star = self.fit(u0, fnorm, n_iter, eta)
        print('Alignment vector : ', u_star, '\n-------------------------------------------------------------')
        Km = np.sum((self.kernels * u_star[:, None, None]), axis=0) ** self.degree
        Km = kernels.normalize_K(Km)
        return Km


def aligned_kernels(methods, lbda, degree, **kwargs):
    data, data1, data2, data3, kernels, ID = utils.get_datas_alignf(methods)
    aligned_k = []
    for k, d in zip(range(3), [data1, data2, data3]):
        X, y, _, _, = d
        Km = NLCK(X, y, ID, kernels, lbda=lbda, eps=1e-9, degree=degree).get_aligned_kernel(**kwargs)
        aligned_k.append(Km)
    return data, data1, data2, data3, aligned_k, ID

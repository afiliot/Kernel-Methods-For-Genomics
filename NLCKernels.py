import numpy as np
from cvxopt import matrix, spmatrix, solvers
import utils
solvers.options['show_progress'] = False
import kernels
from kernels import normalize_K
import SVM


class NLCK():
    """
    Implementation of https://cs.nyu.edu/~mohri/pub/nlk.pdf
    """
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
        self.C = 1 / (2 * self.lbda * self.n)
        self.eps = eps
        self.degree = degree
    
    def svm_step(self, u):

        r = np.arange(self.n)
        o = np.ones(self.n)
        z = np.zeros(self.n)

        K = 0.5 * np.sum((self.kernels_fit * u[:, None, None]), axis=0) ** self.degree

        P = matrix(K.astype(float), tc='d')
        q = matrix(-self.y, tc='d')
        G = spmatrix(np.r_[self.y, -self.y], np.r_[r, r + self.n], np.r_[r, r], tc='d')
        h = matrix(np.r_[o * self.C, z], tc='d')

        # call the solver
        sol = solvers.qp(P, q, G, h)

        # alpha
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

    def get_K(self, u0=0, fnorm=1, n_iter=20, eta=1):
        u_star = self.fit(u0, fnorm, n_iter, eta)
        print('Alignment vector : ', u_star, '\n-------------------------------------------------------------')
        Km = np.sum((self.kernels * u_star[:, None, None]), axis=0) ** self.degree
        Km = kernels.normalize_K(Km)
        return Km


def aligned_kernels(methods, lbdas, degrees, **kwargs):
    data, data1, data2, data3, kernels, ID = utils.get_all_data(methods)
    aligned_k = []
    for d, l, data in zip(degrees, lbdas, [data1, data2, data3]):
        X, y, _, _,_ = data
        Km = NLCK(X, y, ID, kernels, lbda=l, eps=1e-9, degree=d).get_K(**kwargs)
        aligned_k.append(Km)
    return data, data1, data2, data3, aligned_k, ID

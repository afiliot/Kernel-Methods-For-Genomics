import numpy as np
from cvxopt import matrix, spmatrix, solvers
import utils
solvers.options['show_progress'] = False
import kernels
from kernels import normalize_K, center_K
import SVM


class NLCK():
    """
    Implementation of https://cs.nyu.edu/~mohri/pub/nlk.pdf
    """
    def __init__(self, X, y, ID, kernels, C=1e-5, eps=1e-8, degree=2):
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
        r = np.arange(self.n)
        o = np.ones(self.n)
        z = np.zeros(self.n)
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
        u = np.random.normal(0, 5, self.p) / self.p
        u = self.normalize(u, u0, fnorm)
        u = np.array([0 if u[i] < 0 else u[i] for i in range(self.p)])
        score_prev = np.inf
        for k in range(n_iter):
            print('Iteration {}, u={}, score={:0.5f}'.format(k, u, score_prev))
            alpha = self.svm_step(u)
            g = self.grad(u, alpha)
            u_next = u - eta * g
            u_next = self.normalize(u_next, u0, fnorm)
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
        print('Normalizing final kernel...\n-------------------------------------------------------------')
        Km = np.sum((self.kernels * u_star[:, None, None]), axis=0) ** self.degree
        Km = kernels.normalize_K(Km)
        return Km


def aligned_kernels(methods, Cs, degrees, **kwargs):
    data, data1, data2, data3, kernels, ID = utils.get_all_data(methods)
    data_k = [data1, data2, data3]
    aligned_k = []
    for d, C, data in zip(degrees, Cs, data_k):
        X, y, _, _,_ = data
        Km = NLCK(X, y, ID, kernels, C=C, eps=1e-9, degree=d).get_K(**kwargs)
        aligned_k.append(Km)
    return data, data1, data2, data3, aligned_k, ID

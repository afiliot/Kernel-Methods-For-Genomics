from kernels import center_K, normalize_K
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import utils
import SVM

class ALIGNF():
    """
    Implementation of https://cs.nyu.edu/~mohri/pub/align.pdf
    """
    def __init__(self, X, y, ID, kernels):
        self.X = X
        self.y = y.loc[:, 'Bound']
        self.Y = np.outer(self.y, self.y)
        self.ID = ID
        self.kernels = kernels
        self.idx = np.where(np.in1d(self.ID, np.array(self.X.loc[:, 'Id'])))[0]
        self.kernels_fit = [K[self.idx][:, self.idx] for K in self.kernels]
        self.c_kernels_fit = self.center(self.kernels_fit)
        self.p = len(self.kernels)
        self.Nfeval = 1
        self.a = self.get_a().T
        self.M = self.get_M()
        self.u_star = self.get_v()

    def center(self, kernels):
        print('Centering kernels...')
        c_kernels = []
        for K in kernels:
            c_kernels.append(normalize_K(center_K(K)))
        return c_kernels

    def get_a(self):
        print('Computing vector a...')
        a = np.zeros(self.p)
        for i, Kc in enumerate(self.c_kernels_fit):
            a[i] = (Kc*self.Y).sum()
        return a

    def get_M(self):
        print('Computing matrix M...')
        M = np.zeros((self.p, self.p))
        for i, Kc_i in enumerate(self.c_kernels_fit):
            for j, Kc_j in enumerate(self.c_kernels_fit):
                if j >= i :
                    M[i, j] = (Kc_i*Kc_j).sum()
                    M[j, i] = M[i, j]
        return M

    def loss(self, v):
        return np.dot(v.T, np.dot(self.M, v)) - 2*np.dot(v, self.a)

    def jac(self, v):
        return 2*np.dot(self.M, v) - 2*self.a

    def callbackF(self, Xi, Yi=0):
        """
        Print useful information about gradient descent evolution. This function aims at selecting the iteration
        for which the accuracy on the validation set was the best (hence the best vectors alphas).
        :param Xi: np.array, values returned by scipy.minimize at each iteration
        :return: None, update print
        """
        if self.Nfeval == 1:
            self.L = self.loss(Xi)
            print('Iteration {0:2.0f} : loss={1:8.4f}'.format(self.Nfeval, self.L))
        else:
            l_next = self.loss(Xi)
            print('Iteration {0:2.0f} : loss={1:8.4f}, tol={2:8.4f}'
                  .format(self.Nfeval, l_next, abs(self.L - l_next)))
            self.L = l_next
        self.Nfeval += 1

    def get_v(self):
        print('Gradient descent...')
        # initialization
        v0 = np.random.randn(self.p)
        # Gradient descent
        bounds = [[0, float(np.inf)]] * self.p
        res = fmin_l_bfgs_b(self.loss, v0, fprime=self.jac, bounds=bounds, pgtol=1e-6, callback=self.callbackF)
        v_star = res[0]
        return v_star / np.linalg.norm(v_star)

    def get_K(self):
        print('Alignment vector : ', self.u_star, '\n-------------------------------------------------------------')
        Km = np.sum((self.kernels * self.u_star[:, None, None]), axis=0)
        return Km


def aligned_kernels(methods):
    data, data1, data2, data3, kernels, ID = utils.get_all_data(methods)
    aligned_k = []
    for d in [data1, data2, data3]:
        X, y, _, _, _ = d
        aligned_k.append(ALIGNF(X, y, ID, kernels).get_K())
    return data, data1, data2, data3, aligned_k, ID


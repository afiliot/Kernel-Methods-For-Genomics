from kernels import center_K
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import utils

class ALIGNF():
    def __init__(self, X, y, ID, kernels):
        self.X = X
        self.y = y.loc[:, 'Bound']
        self.Y = np.outer(self.y, self.y)
        self.ID = ID
        self.idx = np.where(np.in1d(self.ID, np.array(self.X.loc[:, 'Id'])))[0]
        self.kernels = [K[self.idx][:, self.idx] for K in kernels]
        self.p = len(kernels)
        self.Nfeval = 1
        self.c_kernels = self.center()
        self.a = self.get_a().T
        self.M = self.get_M()

    def center(self):
        print('Centering kernels...')
        c_kernels = []
        for K in self.kernels:
            c_kernels.append(center_K(K))
        return c_kernels

    def get_a(self):
        print('Computing vector a...')
        a = np.zeros(self.p)
        for i, Kc in enumerate(self.c_kernels):
            a[i] = (Kc*self.Y).sum()
        return a

    def get_M(self):
        print('Computing matrix M...')
        M = np.zeros((self.p, self.p))
        for i, Kc_i in enumerate(self.c_kernels):
            for j, Kc_j in enumerate(self.c_kernels):
                if i>=j:
                    M[i, j] = (Kc_i*Kc_j).sum()
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
        v0 = np.zeros(self.p)
        # Gradient descent
        bounds = [[0, float(np.inf)]] * self.p
        res = fmin_l_bfgs_b(self.loss, v0, fprime=self.jac, bounds=bounds, pgtol=1e-6, callback=self.callbackF)
        v_star = res[0]
        return v_star / np.linalg.norm(v_star)

    def get_K(self):
        u_star = self.get_v()
        return u_star


def aligned_kernel(X_train, y_train, ID, kernels):
    alignf = ALIGNF(X_train, y_train, ID, kernels)
    u = alignf.get_K()
    print('Alignment vector : ', u)
    Km = np.sum(u[i] * kernels[i] for i in range(len(kernels)))
    return Km


def aligned_kernels(methods):
    data, data1, data2, data3, kernels, ID = utils.get_datas_alignf(methods)
    aligned_k = []
    for k, data in zip(range(1, 4), [data1, data2, data3]):
        X, y, _, _, = data
        Km = aligned_kernel(X, y, ID, kernels)
        aligned_k.append(Km)
    return data, data1, data2, data3, aligned_k, ID



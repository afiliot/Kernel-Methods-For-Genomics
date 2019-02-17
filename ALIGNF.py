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

    def get_aligned_kernel(self):
        print('Alignment vector : ', self.u_star, '\n-------------------------------------------------------------')
        Km = np.sum((self.kernels * self.u_star[:, None, None]), axis=0)
        return normalize_K(Km)


def aligned_kernels(methods):
    data, data1, data2, data3, kernels, ID = utils.get_datas_alignf(methods)
    aligned_k = []
    for k, d in zip(range(3), [data1, data2, data3]):
        X, y, _, _, = d
        Km = ALIGNF(X, y, ID, kernels).get_aligned_kernel()
        aligned_k.append(Km)
    return data, data1, data2, data3, aligned_k, ID


def export_predictions(data, data1, data2, data3, K, ID, C_opts):
    X_train, y_train, X_val, y_val, X_test = data
    X_train_1, y_train_1, X_val_1, y_val_1 = data1
    X_train_2, y_train_2, X_val_2, y_val_2 = data2
    X_train_3, y_train_3, X_val_3, y_val_3 = data3
    _, _, _, _, X_test_1, _, _ = utils.select_k(1, X_train, y_train, X_val, y_val, X_test, K[0], ID)
    _, _, _, _, X_test_2, _, _ = utils.select_k(2, X_train, y_train, X_val, y_val, X_test, K[1], ID)
    _, _, _, _, X_test_3, _, _ = utils.select_k(3, X_train, y_train, X_val, y_val, X_test, K[2], ID)
    C_opt_1, C_opt_2, C_opt_3 = C_opts
    svm_1 = SVM.C_SVM(K[0], ID, C=C_opt_1, print_callbacks=False)
    res = svm_1.fit(X_train_1, y_train_1)
    pred_tr_1 = svm_1.predict(X_train_1)
    print('Accuracy on train set: {:0.4f}'.format(svm_1.score(pred_tr_1, y_train_1)))
    pred_val_1 = svm_1.predict(X_val_1)
    print('Accuracy on val set: {:0.4f}'.format(svm_1.score(pred_val_1, y_val_1)))
    svm_2 = SVM.C_SVM(K[1], ID, C=C_opt_2, print_callbacks=False)
    res = svm_2.fit(X_train_2, y_train_2)
    pred_tr_2 = svm_2.predict(X_train_2)
    print('Accuracy on train set: {:0.4f}'.format(svm_2.score(pred_tr_2, y_train_2)))
    pred_val_2 = svm_2.predict(X_val_2)
    print('Accuracy on val set: {:0.4f}'.format(svm_2.score(pred_val_2, y_val_2)))
    svm_3 = SVM.C_SVM(K[2], ID, C=C_opt_3, print_callbacks=False)
    res = svm_3.fit(X_train_3, y_train_3)
    pred_tr_3 = svm_3.predict(X_train_3)
    print('Accuracy on train set: {:0.4f}'.format(svm_3.score(pred_tr_3, y_train_3)))
    pred_val_3 = svm_3.predict(X_val_3)
    print('Accuracy on val set: {:0.4f}'.format(svm_3.score(pred_val_3, y_val_3)))
    y_pred_test = utils.export_predictions([svm_1, svm_2, svm_3], [X_test_1, X_test_2, X_test_3])
    return y_pred_test


import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import cvxopt
import cvxopt.solvers


class C_SVM():
    """
    Implementation of C-SVM algorithm
    """
    def __init__(self, K, ID, C=10, eps=1e-5, solver='CVX', print_callbacks=True):
        """
        :param K: np.array, kernel
        :param ID: np.array, Ids (for ordering)
        :param C: float, regularization constant
        :param eps: float, threshold determining whether alpha is a support vector or not
        :param solver: int, choose between 'CVX' or 'BFGS'
        :param print_callbacks: Bool, print evolution of gradient descent when using 'L-BFGS-B' solver (suggested)
        """
        self.K = K
        self.ID = ID
        self.C = C
        self.eps = eps
        self.solver = solver
        self.print_callbacks = print_callbacks
        self.Nfeval = 1

    def loss(self, a):
        """
        :param a: np.array, alphas
        :return: float, loss function
        """
        return -(2 * np.dot(a, self.y_fit) - np.dot(a.T, np.dot(self.K_fit, a)))

    def jac(self, a):
        """
        :param a: np.array, alphas
        :return: np.array, loss Jacobian
        """
        return -(2 * self.y_fit - 2*np.dot(self.K_fit, a))

    def callbackF(self, Xi, Yi=0):
        """
        Print useful information about gradient descent evolution.
        :param Xi: np.array, values returned by scipy.minimize at each iteration
        :return: None, update print
        """
        if self.print_callbacks:
            if self.Nfeval == 1:
                self.L = self.loss(Xi)
                print('Iteration {0:2.0f} : loss={1:8.4f}'.format(self.Nfeval, self.L))
            else:
                l_next = self.loss(Xi)
                print('Iteration {0:2.0f} : loss={1:8.4f}, tol={2:8.4f}'
                      .format(self.Nfeval, l_next, abs(self.L - l_next)))
                self.L = l_next
            self.Nfeval += 1
        else:
            self.Nfeval += 1

    def fit(self, X, y):
        """
        Train C-SVM on X and y.
        :param X: pd.DataFrame, training features
        :param y: pd.DataFrame, training labels
        """
        self.Id_fit = np.array(X.loc[:, 'Id'])
        self.idx_fit = np.array([np.where(self.ID == self.Id_fit[i])[0] for i in range(len(self.Id_fit))]).squeeze()
        self.K_fit = self.K[self.idx_fit][:, self.idx_fit]
        self.y_fit, self.X_fit, = np.array(y.loc[:, 'Bound']), X
        self.n = self.K_fit.shape[0]
        if self.solver == 'BFGS':
            # initialization
            a0 = np.random.randn(self.n)
            # Gradient descent
            bounds_down = [-self.C if self.y_fit[i] <= 0 else 0 for i in range(self.n)]
            bounds_up = [+self.C if self.y_fit[i] >= 0 else 0 for i in range(self.n)]
            bounds = [[bounds_down[i], bounds_up[i]] for i in range(self.n)]
            res = fmin_l_bfgs_b(self.loss, a0, fprime=self.jac, bounds=bounds, callback=self.callbackF)
            self.a = res[0]
        elif self.solver == 'CVX':
            r, o, z = np.arange(self.n), np.ones(self.n), np.zeros(self.n)
            P = cvxopt.matrix(self.K_fit.astype(float), tc='d')
            q = cvxopt.matrix(-self.y_fit, tc='d')
            G = cvxopt.spmatrix(np.r_[self.y_fit, -self.y_fit], np.r_[r, r + self.n], np.r_[r, r], tc='d')
            h = cvxopt.matrix(np.r_[o * self.C, z], tc='d')
            cvxopt.solvers.options['show_progress'] = False
            sol = cvxopt.solvers.qp(P, q, G, h)
            self.a = np.ravel(sol['x'])
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

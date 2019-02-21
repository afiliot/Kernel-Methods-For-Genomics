import numpy as np
import copy

class KLR():
    def __init__(self, K, ID,eps=1e-5, lbda=0.1, tol=1e-5, maxiter=50, solver=None, print_callbacks=True):
        """
        kernel specifies the type of kernel to use
        lamb specifies the regularization parameter
        """
        self.K = K
        self.ID = ID
        self.eps = eps
        self.lbda = lbda
        self.tol = tol
        self.maxiter = maxiter
        self.solver = solver
        self.print_callbacks = print_callbacks
        self.Nfeval = 1

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def IRLS(self, K, y, alpha):
        """
        itertive step to update alpha when training the classifier
        """
        m = np.dot(K, alpha)
        W = self.sigmoid(m) * self.sigmoid(-m)
        z = m + y/self.sigmoid(-y*m)
        return W, z

    def WKRR(self, K, W, z):
        W_s = np.diag(np.sqrt(W))
        A = np.dot(np.dot(W_s, K), W_s) + self.n * self.lbda * np.eye(self.n)
        A = np.dot(np.dot(W_s, np.linalg.inv(A)), W_s)
        return np.dot(A, z)

    def fit(self, X, y):
        self.Id_fit = np.array(X.loc[:, 'Id'])
        self.idx_fit = np.array([np.where(self.ID == self.Id_fit[i])[0] for i in range(len(self.Id_fit))]).squeeze()
        self.K_fit = self.K[self.idx_fit][:, self.idx_fit]
        self.y_fit, self.X_fit, = np.array(y.loc[:, 'Bound']), X
        self.n = self.K_fit.shape[0]
        alpha_prev = np.zeros(self.n)
        diff = np.inf
        for _ in range(self.maxiter):
            if diff > self.tol:
                W, z = self.IRLS(self.K_fit, self.y_fit, alpha_prev)
                alpha = self.WKRR(self.K_fit, W, z)
                diff = np.linalg.norm(alpha-alpha_prev, ord=2)
                alpha_prev = copy.copy(alpha)
        self.a = alpha_prev
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

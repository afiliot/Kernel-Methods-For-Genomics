import numpy as np
import copy

class KLR():
    """
    Implementation of Kernel Logistic Regression
    """
    def __init__(self, K, ID,eps=1e-5, lbda=0.1, tol=1e-5, maxiter=50, solver=None):
        """
        :param K: np.array, kernel
        :param ID: np.array, Ids (for ordering)
        :param eps: float, threshold determining whether alpha is a support vector or not
        :param lbda: float, regularization parameter
        :param tol: float, stopping criteria
        :param maxiter: int, maximum number of iterations for KLR
        :param solver: None
        """
        self.K = K
        self.ID = ID
        self.eps = eps
        self.lbda = lbda
        self.tol = tol
        self.solver = solver
        self.maxiter = maxiter

    def sigmoid(self, x):
        """
        Compute sigma(x) with sigma : x -> 1 / (1+e^(-x))
        """
        return 1/(1+np.exp(-x))

    def IRLS(self, K, y, alpha):
        """
        Iterative step to update alpha when training the classifier
        :param K: np.array, kernel
        :param y: np.array, labels
        :param alpha: np.array
        :return: - W: np.array
                 - z: np.array
        """
        m = np.dot(K, alpha)
        W = self.sigmoid(m) * self.sigmoid(-m)
        z = m + y/self.sigmoid(-y*m)
        return W, z

    def WKRR(self, K, W, z):
        """
        Compute new alpha
        :param K: np.array, kernel
        :param W: np.array
        :param z: np.array
        :return: np.array, new alpha
        """
        W_s = np.diag(np.sqrt(W))
        A = np.dot(np.dot(W_s, K), W_s) + self.n * self.lbda * np.eye(self.n)
        A = np.dot(np.dot(W_s, np.linalg.inv(A)), W_s)
        return np.dot(A, z)

    def fit(self, X, y):
        """
        Train KLR on X and y
        :param X: pd.DataFrame, training features
        :param y: pd.DataFrame, training labels
        """
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


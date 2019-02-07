import numpy as np
from tqdm import tqdm_notebook as tqdm


def beta(d, k):
    """
    Compute beta weights for Weighted Degree Kernel
    :param d: int, maximal degree
    :param k: int, current degree
    :return:
        - beta: float, weight(k, d)
    """
    return 2*(d-k+1)/d/(d+1)


def get_WD_k(x, y, d, L):
    """
    Compute, for two sequences x and y, K(x, y)
    :param x: string, DNA sequence
    :param y: string, DNA sequence
    :param d: int, maximal degree
    :param L: int, length of DNA sequences
    :return:
        - K(x, y): float
    """
    c_t = 0
    for k in range(1, d+1):
        beta_k = beta(d, k)
        c_st = 0
        for l in range(1, L-k+1):
            c_st += (x[l:l+k] == y[l:l+k])
        c_t += beta_k * c_st
    return c_t


def get_WD_K(X, d):
    """
    Compute K(x, y) for each x, y in DNA sequences for Weighted Degree Kernel
    :param: X: pd.DataFrame, features
    :param d: int, maximal degree
    :return:
        - K: np.array, kernel
        - string to describe the method (further used for saving the data)
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        L = len(x)
        K[i, i] = L - 1 + (1-d)/3
        for j, y in enumerate(X.loc[:, 'seq']):
            if j > i:
                K[i, j] = get_WD_k(x, y, d, L)
                K[j, i] = K[i, j]
    return K, 'WD'
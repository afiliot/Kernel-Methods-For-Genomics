import numpy as np
from tqdm import tqdm as tqdm
from itertools import product


################################################# Weight Degree Kernel #################################################

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
    return K

################################################ Mismatch (k,m) Kernel #################################################


def get_phi_km(x, k, m, betas):
    """
    Compute feature vector of sequence x for Mismatch (k,m) Kernel
    :param x: string, DNA sequence
    :param k: int, length of k-mers
    :param m: int, maximal mismatch
    :param betas: list, all combinations of k-mers drawn from 'A', 'C', 'G', 'T'
    :return: np.array, feature vector of x
    """
    phi_km = np.zeros(len(betas))
    for i in range(len(x)-k+1):
        kmer = format(x[i:i+k])
        for i, b in enumerate(betas):
            phi_km[i] = phi_km[i] + 1 if np.sum(kmer != b) <= m else phi_km[i]
    return phi_km


def letter_to_num(x):
    """
    Replace letters by numbers (but still strings)
    :param x: string, DNA sequence
    :return: string
    """
    return x.replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4')


def format(x):
    """
    Transform string 'AGCT' to list [1, 3, 2, 4]
    :param x: string, DNA sequence
    :return: np.array, array of ints with 'A':1, 'C':2, 'G':3, 'T':4
    """
    return np.array(list(letter_to_num(x)), dtype=int)


def get_mismatch_K(X, k, m):
    """
    Compute K(x, y) for each x, y in DNA sequences for Weighted Degree Kernel
    :param: X: pd.DataFrame, features
    :param k: int, length of k-mers
    :param m: int, maximal mismatch
    :return: np.array, kernel
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    betas = [format(''.join(c)) for c in product('ACGT', repeat=k)]
    phi_km_x = []
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Computing feature vectors'):
        x = letter_to_num(x)
        phi_km_x.append(get_phi_km(x, k, m, betas))
    phi_km_x = np.array(phi_km_x)
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        for j, y in enumerate(X.loc[:, 'seq']):
            if j >= i:
                K[i, j] = np.dot(phi_km_x[i], phi_km_x[j])
                K[j, i] = K[i, j]
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Normalizing kernel'):
        for j, y in enumerate(X.loc[:, 'seq']):
            if j > i:
                K[i, j] = K[i, j] / np.sqrt(K[i, i] * K[j, j])
                K[j, i] = K[i, j]
    np.fill_diagonal(K, np.ones(n))
    return K


def select_method(X, method, param):
    """
    Given method and param dictionary, compute kernel
    :param X: pd.DataFrame, features
    :param method: string, method to apply for building the kernel
    :param param: dict, dictionary mapping method to parameters
    :return:
    """
    if method[:2] == 'WD':
        p = param[method]
        K = get_WD_K(X, p)
    if method[:2] == 'MM':
        k, m = int(method[2]), int(method[3])
        K = get_mismatch_K(X, k, m)
    return K



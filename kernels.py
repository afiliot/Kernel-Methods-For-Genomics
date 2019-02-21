import numpy as np
from tqdm import tqdm as tqdm
from itertools import product
from copy import deepcopy
from scipy.sparse.linalg import eigs
from numpy.linalg import multi_dot
from numba import jit

################################################### Spectrum Kernel ####################################################


def get_phi_u(x, k, betas):
    """
    Compute feature vector of sequence x for Spectrum (k) Kernel
    :param x: string, DNA sequence
    :param k: int, length of k-mers
    :param betas: list, all combinations of k-mers drawn from 'A', 'C', 'G', 'T'
    :return: np.array, feature vector of x
    """
    phi_u = np.zeros(len(betas))
    for i in range(len(x) - k + 1):
        kmer = x[i:i + k]
        for i, b in enumerate(betas):
            phi_u[i] += (b == kmer)
    return phi_u


def get_spectrum_K(X, k):
    """
    Compute K(x, y) for each x, y in DNA sequences for Spectrum Kernel
    :param: X: pd.DataFrame, features
    :param k: int, length of k-mers
    :return: np.array, kernel
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    betas = [''.join(c) for c in product('ACGT', repeat=k)]
    phi_u = []
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Computing feature vectors'):
        phi_u.append(get_phi_u(x, k, betas))
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        for j, y in enumerate(X.loc[:, 'seq']):
            if j >= i:
                K[i, j] = np.dot(phi_u[i], phi_u[j])
                K[j, i] = K[i, j]
    K = K
    return K


################################################# Weight Degree Kernel #################################################


def beta(d, k):
    """
    Compute beta weights for Weighted Degree Kernel
    :param d: int, maximal degree
    :param k: int, current degree
    :return:
        - beta: float, weight(k, d)
    """
    return 2 * (d - k + 1) / d / (d + 1)


def get_WD_d(x, y, d, L):
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
    for k in range(1, d + 1):
        beta_k = beta(d, k)
        c_st = 0
        for l in range(1, L - k + 1):
            c_st += (x[l:l + k] == y[l:l + k])
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
        K[i, i] = L - 1 + (1 - d) / 3
        for j, y in enumerate(X.loc[:, 'seq']):
            if j > i:
                K[i, j] = get_WD_d(x, y, d, L)
                K[j, i] = K[i, j]
    return K

########################################### Weight Degree Kernel with Shifts ###########################################


def delta(s):
    """
    Compute delta coefficients for weight degree kernel with shifts
    :param s: int
    :return: delta(s)
    """
    return 1/2/(s+1)


def get_WDShifts_d(x, y, d, S, L):
    """
    Compute, for two sequences x and y, K(x, y)
    :param x: string, DNA sequence
    :param y: string, DNA sequence
    :param d: int, maximal degree
    :param S: int, maximal shift
    :param L: int, length of DNA sequences
    :return:
        - K(x, y): float
    """
    c_t = 0
    for k in range(1, d + 1):
        beta_k = beta(d, k)
        c_st = 0
        for i in range(1, L - k + 1):
            for s in range(0, S+1):
                if s+i < L:
                    c_st += delta(s) * ((x[i+s:i+s+k] == y[i:i+k]) + (x[i:i+k] == y[i+s:i+s+k]))
        c_t += beta_k * c_st
    return c_t


def get_WDShifts_K(X, d, S):
    """
    Compute K(x, y) for each x, y in DNA sequences for Weighted Degree Kernel
    :param: X: pd.DataFrame, features
    :param d: int, maximal degree
    :param S: int, maximal shift
    :return:
        - K: np.array, kernel
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        L = len(x)
        for j, y in enumerate(X.loc[:, 'seq']):
            if j >= i:
                K[i, j] = get_WDShifts_d(x, y, d, S, L)
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
    for i in range(101 - k + 1):
        kmer = x[i:i + k]
        for i, b in enumerate(betas):
            phi_km[i] += (np.sum(kmer != b) <= m)
    return phi_km


def letter_to_num(x):
    return x.replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4')


def format(x):
    """
    Transform string 'AGCT' to list [1, 3, 2, 4]
    :param x: string, DNA sequence
    :return: np.array, array of ints with 'A':1, 'C':2, 'G':3, 'T':4
    """
    return np.array(list(letter_to_num(x))).astype(int)


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
    betas = np.array([format(''.join(c)) for c in product('ACGT', repeat=k)])
    phi_km_x = np.zeros((n, len(betas)))
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Computing feature vectors'):
        x = format(x)
        phi_km_x[i] = get_phi_km(x, k, m, betas)
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        for j, y in enumerate(X.loc[:, 'seq']):
            if j >= i:
                K[i, j] = np.dot(phi_km_x[i], phi_km_x[j])
                K[j, i] = K[i, j]
    K = normalize_K(K)
    return K


################################################ Local Alignment Kernel ################################################

# substitution matrix extracted from BLOSUM62
S = np.array([[4, 0, 0, 0], [0, 9, -3, -1], [0, -3, 6, 2], [0, -1, -2, 5]])


def affine_align(x, y, e, d, beta):
    """
    Implement local alignment kernel where g(n) is linear g(0)=0 and g(n)=e+d(n-1)
    :param x: string, DNA sequence
    :param y: string, DNA sequence
    :param e: int, gap open penalty
    :param d: int, gap extend penalty
    :param beta: float, parameter for k_beta(x, y)
    :return: float, K_beta(x, y)
    """
    x, y = format(x)-1, format(y)-1
    n_x, n_y = len(x), len(y)
    M, X, Y, X2, Y2 = [np.zeros((n_x + 1, n_y + 1))]*5
    for i in range(1, n_x):
        for j in range(1, n_y):
            M[i, j] = np.exp(beta * S[x[i], y[j]]) * (1 + X[i-1, j-1] + Y[i-1, j-1] + M[i-1, j-1])
            X[i, j] = np.exp(beta * d) * M[i-1, j] + np.exp(beta * e) * X[i-1, j]
            Y[i, j] = np.exp(beta * d) * (M[i, j-1] + X[i, j-1]) + np.exp(beta * e) * Y[i, j-1]
            X2[i, j] = M[i-1, j] + X2[i-1, j]
            Y2[i, j] = M[i, j-1] + X2[i, j-1] + Y2[i, j-1]
    return (1/beta) * np.log(1 + X2[n_x, n_y] + Y2[n_x, n_y] + M[n_x, n_y])


def Smith_Waterman(x, y, e=11, d=1, beta=0.5):
    """
    Implement Smith Waterman algorithm for computing Local Alignment Kernel (LA). The difference here is that
    we consider the maximum between the terms and no more sums.
    :param x: string, DNA sequence
    :param y: string, DNA sequence
    :param e: int, gap open penalty
    :param d: int, gap extend penalty
    :param beta: float, parameter for k_beta(x, y)
    :return: float, K_beta(x, y)
    """
    x, y = format(x) - 1, format(y) - 1
    n_x, n_y = len(x), len(y)
    M, X, Y, X2, Y2 = [np.zeros((n_x + 1, n_y + 1))] * 5
    for i in range(1, n_x):
        for j in range(1, n_y):
            M[i, j] = np.exp(beta * S[x[i], y[j]]) * max(1, X[i - 1, j - 1], Y[i - 1, j - 1], M[i - 1, j - 1])
            X[i, j] = max(np.exp(beta * d) * M[i - 1, j], np.exp(beta * e) * X[i - 1, j])
            Y[i, j] = max(np.exp(beta * d) * M[i, j - 1], np.exp(beta * d) * X[i, j - 1], np.exp(beta * e) * Y[i, j - 1])
            X2[i, j] = max(M[i - 1, j], X2[i - 1, j])
            Y2[i, j] = max(M[i, j - 1], X2[i, j - 1], Y2[i, j - 1])
    return (1/beta) * np.log(max(1, X2[n_x, n_y], Y2[n_x, n_y], M[n_x, n_y]))


def get_LA_K(X, e=11, d=1, beta=0.5, smith=0, eig=1):
    """
    Compute Local Alignment Kernel
    :param X: pd.DataFrame, features
    :param e: int, gap open penalty
    :param d: int, gap extend penalty
    :param beta: float, parameter for k_beta(x, y)
    :param smith: 0 or 1, whether to use or not Smith algorithm
    :param eig: 0 or 1, whether to substract the smallest negative eigenvalue from K after computation (to make K p.s.d)
    If eig=0 then empirical kernel method is used.
    :return: np.array, kernel
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        for j, y in enumerate(X.loc[:, 'seq']):
            if j >= i:
                K[i, j] = Smith_Waterman(x, y, e, d, beta) if smith else affine_align(x, y, e, d, beta)
                K[j, i] = K[i, j]
    K1 = deepcopy(K)
    if eig == 1:
        vp = np.min(np.real(eigs(K1)[0]))
        s = vp if vp < 0 else 0
        np.fill_diagonal(K1, np.diag(K1) - s * np.ones(n))
    else:
        for i in tqdm(range(K1.shape[0]), desc='Empirical kernel'):
            for j in range(i, n):
                K1[i, j] = np.dot(K[i], K[j])
                K1[j, i] = K1[i, j]
    return K


#################################################### String Kernel #####################################################


def rec(func):
    """
    method for handling recursion memory
    """
    memory = {}
    def recd(*args):
        key = '-'.join('[%s]' % arg for arg in args)
        if key not in memory:
            memory[key] = func(*args)
        return memory[key]
    return recd


@rec
def B_k(lbda, k, x, y):
    """
    Recursively compute B_k
    :param lbda: float
    :param k: int, length of k-mers
    :param x: string, DNA sequence
    :param y: string, DNA sequence
    """
    if k == 0:
        return 1
    n_x, n_y = len(x), len(y)
    if n_x < k or n_y < k:
        return 0
    sub_x, sub_y = x[:-1], y[:-1]
    return (
            lbda * B_k(lbda, k, sub_x, y)
            + lbda * B_k(lbda, k, x, sub_y)
            - (lbda**2) * B_k(lbda, k, sub_x, sub_y)
            + ((lbda**2) * B_k(lbda, k-1, sub_x, sub_y) if x[-1] == y[-1] else 0)
           )

@rec
def K_k(lbda, k, x, y):
    """
    Recursively compute K_k
    :param lbda: float
    :param k: int, length of k-mers
    :param x: string, DNA sequence
    :param y: string, DNA sequence
    :return K(x, y)
    """
    if k == 0:
        return 1
    n_x, n_y = len(x), len(y)
    if n_x < k or n_y < k:
        return 0
    sub_x = x[:-1]
    a = x[-1]
    return (
            K_k(lbda, k, sub_x, y)
            + (lbda**2) * sum(B_k(lbda, k-1, sub_x, y[:j]) for j in range(n_y) if y[j] == a)
           )


def get_substring_K(X, lbda, k):
    """
    Compute Substring Kernel (lbda, k)
    :param X: pd.DataFrame, features
    :param lbda: float
    :param k: int, length of k-mers
    :return: np.array, kernel
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        for j, y in enumerate(X.loc[:, 'seq']):
            print(j)
            if j >= i:
                K[i, j] = K_k(lbda, k, x, y)
                K[j, i] = K[i, j]
    return K


###################################################### Normalize #######################################################

def center_K(K):
    """
    Normalize kernel
    :param K: np.array
    :return: np.array
    """
    n = K.shape[0]
    B = np.eye(n) - np.ones((n, n))/n
    return multi_dot([B, K, B])


def normalize_K(K):
    """
    Normalize kernel
    :param K: np.array
    :return: np.array
    """
    if K[0, 0] == 1:
        print('Kernel already normalized')
    else:
        n = K.shape[0]
        diag = np.sqrt(np.diag(K))
        for i in range(n):
            d = diag[i]
            for j in range(i+1, n):
                K[i, j] /= (d * diag[j])
                K[j, i] = K[i, j]
        np.fill_diagonal(K, np.ones(n))
    return K

#################################################### Select method #####################################################


def select_method(X, method):
    """
    Given method and param dictionary, compute kernel
    :param X: pd.DataFrame, features
    :param method: string, method to apply for building the kernel
    :return: np.array, K
    """
    if method[:2] == 'SP':
        k = int(method[2:])
        K = get_spectrum_K(X, k)
    elif method[:2] == 'WD' and method[2] != 'S':
        k = int(method[2:])
        K = get_WD_K(X, k)
    elif method[:2] == 'MM':
        k, m = int(method[2]), int(method[3])
        K = get_mismatch_K(X, k, m)
    elif method[:2] == 'LA':
        m = method.split('_')
        e, d, beta = [float(m[i][1:]) for i in range(1, 4)]
        smith, eig = int(m[4][5:]), int(m[5][3:])
        K = get_LA_K(X, e, d, beta, smith, eig)
    elif method[:3] == 'WDS':
        m = method.split('_')
        d, S = int(m[1][1:]), int(m[2][1:])
        K = get_WDShifts_K(X, d, S)
    elif method[:2] == 'SS':
        m = method.split('_')
        lbda, k = float(m[1][1:]), int(m[2][1:])
        K = get_substring_K(X, lbda, k)
    return K

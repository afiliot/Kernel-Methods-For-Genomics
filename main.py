import utils
from SVM import C_SVM
from ALIGNF import ALIGNF, aligned_kernels
from NLCKernels import NLCK, aligned_kernels
import numpy as np
from itertools import product

check_alignf = False
check_NLCK = False
check_method = True
build_kernel = False

if __name__ == '__main__':
    # Build kernels
    # Methods available:
    # - SPx : spectrum kernel with x = int
    # - WDx : Weight degree kernel with x = int
    # - WDS_dx_sy : Weight degree kernel with x = int (d), y = int (s)
    # - MMxy: mismatch kernel with x = int (k) and y = int (m)
    # - LA_ex_dy_bz_smithX_eigY with x = int (e), y = int (d), z = float (beta), X = 0/1 (smith), Y = 0/1 (eig)

    if build_kernel:
        X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method='SP7', replace=True)

    elif check_method:
        method = 'SP6'
        data, data1, data2, data3, K, ID = utils.get_all_data([method])
        svm = C_SVM(K, ID, solver='BFGS')
        X_train_1, y_train_1, X_val_1, y_val_1, X_test_1 = data1
        Cs = np.sort([i * 10 ** j for (i, j) in product(range(1, 10), range(-5, 2))])
        utils.cross_validation(Ps=Cs, data=data1, algo='KRR', kfolds=3, pickleName='cv_C_SVM_f1', K=K, ID=ID)

    elif check_alignf:
        methods = ['SP6', 'WD5', 'WD4', 'SP5']
        data, data1, data2, data3, kernels, ID = aligned_kernels(methods)
        K = kernels[0]  # first data set
        svm = C_SVM(K, ID)
        X_train_1, y_train_1, X_val_1, y_val_1, X_test_1 = data1
        svm.fit(X_train_1, y_train_1)
        Cs = np.sort([i * 10 ** j for (i, j) in product(range(1, 10), range(-5, 2))])
        utils.cross_validation(Ps=Cs, data=data1, algo='CSVM', kfolds=5, pickleName='cv_C_SVM_f1', K=K, ID=ID)

    elif check_NLCK:
        methods = ['SP6', 'MM61', 'WD10']
        lbdas = [0.01, 0.01, 0.01]
        degrees = [3, 2, 2]
        data, data1, data2, data3, kernels, ID = aligned_kernels(methods, lbdas=lbdas, degrees=degrees)
        K = kernels[0]  # first data set
        svm = C_SVM(kernels[0], ID)
        X_train_1, y_train_1, X_val_1, y_val_1, X_test_1 = data1
        Cs = np.sort([i * 10 ** j for (i, j) in product(range(1, 10), range(-5, 2))])
        utils.cross_validation(Ps=Cs, data=data1, algo='CSVM', kfolds=5, pickleName='cv_C_SVM_f1', K=K, ID=ID)













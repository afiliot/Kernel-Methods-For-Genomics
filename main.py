import utils
import SVM
import ALIGNF
import NLCKernels
import numpy as np
from itertools import product
import kernels as ke

check_alignf = False
check_NLCK = False
check_method = False
build_kernel = True

if __name__ == '__main__':
    # Build kernels
    # Methods available:
    # - SPx : spectrum kernel with x = int
    # - WDx : Weight degree kernel with x = int
    # - WDS_dx_sy : Weight degree kernel with x = int (d), y = int (s)
    # - MMxy: mismatch kernel with x = int (k) and y = int (m)
    # - LA_ex_dy_bz_smithX_eigY with x = int (e), y = int (d), z = float (beta), X = 0/1 (smith), Y = 0/1 (eig)

    if check_alignf:
        methods = ['SP6', 'WD5', 'WD4', 'SP5']
        data, data1, data2, data3, kernels, ID = ALIGNF.aligned_kernels(methods)
        svm = SVM.C_SVM(kernels[0], ID)
        X_train_1, y_train_1, X_val_1, y_val_1 = data1
        svm.fit(X_train_1, y_train_1)
        Cs = np.sort([i * 10 ** j for (i, j) in product(range(1, 10), range(-5, 2))])
        SVM.cv(Cs=Cs, data=data1,
               kfolds=5, pickleName='cv_C_SVM_f1', K=kernels[0], ID=ID)

    elif check_method:
        methods = ['SP6']
        utils.run_expe(methods, algo='KRR')

    elif build_kernel:
        X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method='SS_l0.5_k5', replace=True)


    elif check_NLCK:
        methods = ['SP6', 'MM61', 'WD10'] #, 'WD4', 'SP5']
        data, data1, data2, data3, kernels, ID = NLCKernels.aligned_kernels(methods, lbda=0.001, degree=2)
        svm = SVM.C_SVM(kernels[0], ID)
        X_train_1, y_train_1, X_val_1, y_val_1 = data1
        Cs = np.sort([i * 10 ** j for (i, j) in product(range(1, 10), range(-5, 2))])
        SVM.cv(Cs=Cs, data=data1, kfolds=5, pickleName='cv_C_SVM_f1', K=kernels[0], ID=ID)













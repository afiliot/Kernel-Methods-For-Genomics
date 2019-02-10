import utils
import SVM

param = {'WD1': 1, 'WD2': 2, 'WD3': 3, 'WD4': 4, 'WD5': 5}

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method="WD3", param=param, replace=False)
    X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, K_1, id_1 = utils.select_k(1, X_train, y_train, X_val, y_val, X_test, K, ID)
    svm = SVM.C_SVM(K, ID, C=0.01, maxiter=20, method='SLSQP')
    svm.fit(X_train_1, y_train_1)
    pred = svm.predict(X_val_1)
    score = svm.score(pred, y_val_1)







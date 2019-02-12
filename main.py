import utils

param = {'WD1': 1, 'WD2': 2, 'WD3': 3, 'WD4': 4, 'WD5': 5, 'WD10': 10}


if __name__ == '__main__':
    # Build kernels
    X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method="MM31", param=param, replace=True)
    X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method="MM41", param=param, replace=True)
    X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method="MM51", param=param, replace=True)
    X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method="MM61", param=param, replace=True)







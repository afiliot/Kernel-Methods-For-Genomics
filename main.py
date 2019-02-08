import utils

param = {'WD1': 1, 'WD2': 2, 'WD3': 3, 'WD4': 4, 'WD5': 5}

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, K = utils.get_training_datas(method="WD3", param=param, replace=True)


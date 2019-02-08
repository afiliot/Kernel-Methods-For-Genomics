import utils

param = {'WD1': 1, 'WD2': 2, 'WD3': 3, 'WD4': 4, 'WD5': 5}

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method="WD3", param=param, replace=True)
    X_train_, y_train_, X_val_, y_val_, X_test_, K_, id_k = utils.select_k(1, X_train, y_train, X_val, y_val, X_test, K, ID)


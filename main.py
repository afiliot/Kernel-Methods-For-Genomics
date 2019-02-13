import utils



if __name__ == '__main__':
    # Build kernels
    X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method='SP3', replace=True)
    X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method='SP4', replace=True)
    X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method='SP5', replace=True)
    X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method='LA_e8_d1', replace=True)
    X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method='LA_e11_d1', replace=True)
    X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method='LA_e5_d2', replace=True)








import utils



if __name__ == '__main__':
    # Build kernels
    # Methods available:
    # - SPx : spectrum kernel with x = int
    # - WDx : Weight degree kernel with x = int
    # - WDS_dx,sy : Weight degree kernel with x = int (d), y = int (s)
    # - MMxy: mismatch kernel with x = int (k) and y = int (m)
    # - LA_ex_dy_bz_smithX_eigY with x = int (e), y = int (d), z = float (beta), X = 0/1 (smith), Y = 0/1 (eig)
    X_train, y_train, X_val, y_val, X_test, K, ID = utils.get_training_datas(method='SP3', replace=True)








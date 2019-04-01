from utils import get_all_data, reformat_data, export_predictions
from NLCKernels import NLCK
from SVM import C_SVM

# Best submission : Non-Linear Combination of 10 kernels listed below:
#methods = ['SP4', 'SP5', 'SP6', 'MM31', 'MM41', 'MM51', 'MM61', 'WD4', 'WD5', 'WD10']
methods = ['MM_k2_m1', 'SP_k1']

# Import data
data, data1, data2, data3, kernels, ID = get_all_data(methods)

# Use the algorithm on the first data set with the corresponding hyperparameters (see the report, table 1)
print('\n\n')
X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, kernels_1, ID_1 = reformat_data(data1, kernels, ID)
Km1 = NLCK(X_train_1, y_train_1, ID_1, kernels_1, C=1, eps=1e-9, degree=3).get_K(fnorm=5, n_iter=50)
svm1 = C_SVM(Km1, ID_1, C=1.9, solver='CVX')
svm1.fit(X_train_1, y_train_1)

# Use the algorithm on the second data set with the corresponding hyperparameters (see the report, table 1)
print('\n\n')
X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, kernels_2, ID_2 = reformat_data(data2, kernels, ID)
Km2 = NLCK(X_train_2, y_train_2, ID_2, kernels_2, C=10, eps=1e-9, degree=4).get_K(fnorm=5, n_iter=50)
svm2 = C_SVM(Km2, ID_2, C=2.1, solver='CVX')
svm2.fit(X_train_2, y_train_2)

# Use the algorithm on the third data set with the corresponding hyperparameters (see the report, table 1)
print('\n\n')
X_train_3, y_train_3, X_val_3, y_val_3, X_test_3, kernels_3, ID_3 = reformat_data(data3, kernels, ID)
Km3 = NLCK(X_train_3, y_train_3, ID_3, kernels_3, C=1e-2, eps=1e-9, degree=3).get_K(fnorm=1, n_iter=50)
svm3 = C_SVM(Km3, ID_3, C=3, solver='CVX')
svm3.fit(X_train_3, y_train_3)

# See scores on validation set
print('\n\nAccuracy on validation set 1: {:0.4f}'.format(svm1.score(svm1.predict(X_val_1), y_val_1)))
print('Accuracy on validation set 2: {:0.4f}'.format(svm2.score(svm2.predict(X_val_2), y_val_2)))
print('Accuracy on validation set 3: {:0.4f}'.format(svm3.score(svm3.predict(X_val_3), y_val_3)))

# Compute predictions
y_pred = export_predictions([svm1, svm2, svm3], [X_test_1, X_test_2, X_test_3])
print('\n\nPredictions ok')
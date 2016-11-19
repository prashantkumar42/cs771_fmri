from sklearn import linear_model
import time
import numpy as np
import scipy.io

print("getting data from fmri_words.mat...")
mat = scipy.io.loadmat('./fMRI/fmri_words.mat')

x_train = mat['X_train']
y_train = mat['Y_train']
x_test = mat['X_test']
y_test = mat['Y_test']
word_features_std = mat['word_features_std']

t = time.time()
ans = word_features_std[np.ix_(((y_train.transpose())[0]) - 1, np.arange(218))]


def fun(alpha_cur, l1_ratio):
    # print("fun called for alpha: " + str(alpha_cur) + " & l1_ratio: " + str(l1_ratio))
    # enet_ = linear_model.ElasticNet(alpha=alpha_cur, l1_ratio=l1_ratio, normalize=True)
    enet_ = linear_model.ElasticNet(alpha=alpha_cur, l1_ratio=l1_ratio)
    enet_.fit(x_train, ans)
    W = clf.coef_
    # print("Weight Vector Calculated...")
    y_feat_test_pred = np.mat(x_test) * np.mat(W.transpose())

    # print("Predicting the labels")
    y_test_pred = np.zeros(60)
    correct_count = 60
    for i in range(60):
        dist_ = np.zeros(2)
        dist_[0] = np.linalg.norm(y_feat_test_pred[i, :] - word_features_std[y_test[i][0] - 1, :])
        dist_[1] = np.linalg.norm(y_feat_test_pred[i, :] - word_features_std[y_test[i][1] - 1, :])
        min_ind = np.argmin(dist_)
        y_test_pred[i] = y_test[i, min_ind]
        correct_count -= min_ind
    # print(y_test_pred[i])
    # print("correct count: " + str(correct_count))
    print("alpha: " + str(alpha_cur) + " l1_ratio: " + str(l1_ratio) + " accuracy: " + str(correct_count * 100 / 60.0) + "%")
    # print("fun exited")

l1_ratios = np.arange(0, 1.1, 0.1)  # generating values for l1 ration used in enet regularisation
l1_ratios = [1]
alpha_enet = [1e-3, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 1, 5, 10, 20]

for alpha_cur in alpha_lasso:
    for l1_ratio in l1_ratios:
        fun(alpha_cur, l1_ratio)

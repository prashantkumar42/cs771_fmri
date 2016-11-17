from sklearn import linear_model
import time
import threading
# import _thread
import numpy as np
import scipy.io

print("getting data from fmri_words.mat...")
mat = scipy.io.loadmat('fmri_words.mat')

x_train = mat['X_train']
y_train = mat['Y_train']
x_test = mat['X_test']
y_test = mat['Y_test']
word_features_std = mat['word_features_std']

t = time.time()
ans = word_features_std[np.ix_(((y_train.transpose())[0]) - 1, np.arange(218))]


# t_diff = time.time()-t
# print(t_diff)

def fun(alpha_cur):
    print("fun called ")
    clf = linear_model.Lasso(alpha=alpha_cur)
    # y=ans[:, 0]
    # clf.fit(x_train,y)
    # print("fitting the train data...")
    clf.fit(x_train, ans)
    # print(clf.coef_.shape)
    # print(np.count_nonzero(clf.coef_))
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
    print("alpha: " + str(alpha_cur) + " accuracy: " + str(correct_count * 100 / 60.0) + "%")
    print("fun exited")


# alpha_lasso = [0.03, 0.04, 0.045, 0.055, 0.06, 0.07, 0.08, 0.09]
alpha_lasso = [0.03, 0.04, 0.045, 0.055]  # , 0.06, 0.07, 0.08, 0.09]
for alpha_cur in alpha_lasso:
    time.sleep(1)
    t = threading.Thread(target=fun, args=(alpha_cur,))
    t.daemon = True
    t.start()
    while 1:
        time.sleep(0.1)

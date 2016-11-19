import scipy.io as lod
import numpy as np
import time
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 100, 50

dict_mat = lod.loadmat("./fMRI/fmri_words.mat")
# dict_mat = lod.loadmat("fmri_words.mat")
X_train = dict_mat['X_train']
Y_train = dict_mat['Y_train']
X_test = dict_mat['X_test']
Y_test = dict_mat['Y_test']
word_features_std = dict_mat['word_features_std']
print type(X_train), type(Y_train), X_train.shape, Y_train.shape


def enet_regression(data, y_val, alpha, l1_ratio ):
    # Fit the model
    enetreg = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio,normalize=True)
    enetreg.fit(data, y_val)
    W = enetreg.coef_

    return W

def accuracy(y_feat_test_pred , Y_test):
    correct_count = 60
    for i in range(60):
        dist_ = np.zeros(2)
        dist_[0] = np.linalg.norm(y_feat_test_pred[i, :] - word_features_std[Y_test[i][0] - 1, :])
        dist_[1] = np.linalg.norm(y_feat_test_pred[i, :] - word_features_std[Y_test[i][1] - 1, :])
        min_ind = np.argmin(dist_)
        # y_test_pred[i] = Y_test[i, min_ind]
        correct_count -= min_ind
        # print(y_test_pred[i])
    print("correct count: " + str(correct_count))
    print("accuracy: " + str(correct_count * 100 / 60.0) + "%")
    return str(correct_count * 100 / 60.0)


# mapping the y_value to the 218 feature
ans = word_features_std[np.ix_(((Y_train.transpose())[0]) - 1, np.arange(218))]

# clf = linear_model.Ridge(alpha=0.1)
# clf.fit(X_train, ans)
l1_ratio = np.arange(0, 1.1, 0.1)  # generating values for l1 ration used in enet regularisation
alpha_enet = [1e-3, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 1, 5, 10, 20]
acc = []
for l1_val in l1_ratio:
    for alpha in alpha_enet:
        print "alpha = ", alpha ,"L1_ratio = " ,l1_val
        t = time.time()
        W = enet_regression(X_train, ans, alpha, l1_val)

        # print " (clf.coef_.shape) = ", (clf.coef_.shape)
        # W = clf.coef_

        y_feat_test_pred = np.mat(X_test) * np.mat(W.transpose())
        acc = acc + [accuracy(y_feat_test_pred,Y_test)]

        # calculating time taken
        t_diff = time.time() - t
        print " t_diff = ", (t_diff)
print alpha_enet,"\n",acc
plt.plot(alpha_enet,acc,'*')
plt.show()
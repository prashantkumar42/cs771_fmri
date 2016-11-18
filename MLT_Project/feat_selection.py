import numpy as np
import scipy.io as data
dict = data.loadmat("C:\Users\BHANU YADAV\Documents\MLT_Project\fmri_words.mat")
X_train = np.array(dict["X_train"])
Y_train = np.array(dict["Y_train"])
X_test = np.array(dict["X_test"])
Y_test = np.array(dict["Y_test"])
word_features_std = np.array(dict["word_features_std"])
X_train
print word_features_std
for i in (1, 300):
    S(i) = word_features_std(Y_train(i))
end
for i in (1, 218):
	for j in (1, 21764):
		B = np.corrcoef(X_train(all, j), S(all, i))
		C(i, j) = B(1, 2)

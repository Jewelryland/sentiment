import numpy as np
from sklearn.linear_model import LogisticRegression as LogRegr

print "Training..."

feats_train = np.load("feats_train.npy").item(0)
feats_test = np.load("feats_test.npy").item(0)
labels_train = np.load("labels_train.npy")
labels_test = np.load("labels_test.npy")

clf = LogRegr()
clf.fit(feats_train, labels_train)
print clf.score(feats_test, labels_test)

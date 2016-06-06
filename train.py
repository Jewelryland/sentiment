import numpy as np
from glob import glob
from sklearn.linear_model import LogisticRegression as LogRegr
from sklearn.grid_search import GridSearchCV

for path in glob("feats/*.npz"):
    name = path.split("/")[-1].split(".")[0]
    max_gram, dim = name.split("_")
    print "Loading 1-%s gram, feature dimension %s" % (max_gram, dim)
    with open(path, "rb") as fin:
        data = np.load(fin)
        feats_train = data["feats_train"].item(0)
        feats_test = data["feats_test"].item(0)
        labels_train = data["labels_train"]
        labels_test = data["labels_test"]

    print "Training..."
    c_grid = 10.0 ** np.arange(-5, 6)
    clf = GridSearchCV(LogRegr(), {'C':c_grid}, n_jobs=-1, cv=5)
    clf.fit(feats_train, labels_train)
    print "train accuracy %f, test accuracy %f" % (clf.score(feats_train, labels_train), clf.score(feats_test, labels_test))

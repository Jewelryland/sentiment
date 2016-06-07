import numpy as np
import cPickle as pickle
from glob import glob
from sklearn.linear_model import LogisticRegression as LogRegr
from sklearn.grid_search import GridSearchCV

def train_models(paths, save_model=False):
    for path in paths:
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

        if save_model:
            with open("models/clf_%s_%s.pkl" % (max_gram, dim), "wb") as fout:
                pickle.dump(clf, fout, -1)

train_models(glob("feats/*.npz"))
# train_models(["feats/3_30000.npz"], True)

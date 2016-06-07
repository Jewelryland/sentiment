import numpy as np
import re
import cPickle as pickle
import nltk
from bs4 import BeautifulSoup
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
from preproc import preproc

def read_reviews(path):
    file_paths = glob(path + "/*.txt")
    reviews = []
    for path in file_paths:
        with open(path, "r") as fin:
            reviews.append(fin.read())
    return reviews

def read_data(path):
    pos_reviews = read_reviews(path + "/pos")
    neg_reviews = read_reviews(path + "/neg")
    reviews = pos_reviews + neg_reviews
    labels = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
    return reviews, labels

print "read data"

TRAIN_DIR = "data/train"
reviews_train, labels_train = read_data(TRAIN_DIR)

reviews_unsup = read_reviews(TRAIN_DIR + "/unsup")

TEST_DIR = "data/test"
reviews_test, labels_test = read_data(TEST_DIR)

print "clean data"

reviews_train = [preproc(review) for review in reviews_train]
reviews_test = [preproc(review) for review in reviews_test]
reviews_unsup = [preproc(review) for review in reviews_unsup]

def extract(max_gram, feat_dims, save_model=False):
    print "extract feature"

    vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features=None, 
            ngram_range=(1, max_gram), sublinear_tf = True )

    vectorizer = vectorizer.fit(reviews_train + reviews_unsup)
    feats_train_ori = vectorizer.transform(reviews_train)
    feats_test_ori = vectorizer.transform(reviews_test)
    print "size of orginal train features", feats_train_ori.shape

    for feat_dim in feat_dims:
        print "perform feature selection"

        fselect = SelectKBest(chi2 , k=feat_dim)
        feats_train = fselect.fit_transform(feats_train_ori, labels_train)
        feats_test = fselect.transform(feats_test_ori)

        print "save features"
        np.savez("feats/%d_%d.npz" % (max_gram, feat_dim), 
                feats_train=feats_train, feats_test=feats_test, 
                labels_train=labels_train, labels_test=labels_test)

        if save_model:
            print "save models"
            with open("models/vectorizer_%d.pkl" % max_gram, "wb") as fout:
                pickle.dump(vectorizer, fout, -1)

            with open("models/fselect_%d_%d.pkl" % (max_gram, feat_dim), "wb") as fout:
                pickle.dump(fselect, fout, -1)

extract(4, [1000, 3000, 10000, 30000, 70000, 100000])
extract(3, [1000, 3000, 10000, 30000, 70000, 100000])
extract(2, [1000, 3000, 10000, 30000, 70000, 100000])
extract(1, [1000, 3000, 10000, 30000, 70000, 100000])
# extract(3, [30000], True)

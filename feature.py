import numpy as np
import re
from bs4 import BeautifulSoup
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2

def read_reviews(path):
    file_paths = glob(path + "/*.txt")
    reviews = []
    for path in file_paths:
        with open(path, "r") as fin:
            reviews.append(fin.read())
    return reviews

def preproc(review):
    review_text = BeautifulSoup(review, "lxml").get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    # stops = set(stopwords.words("english"))
    # words = [w for w in words if not w in stops]

    # b=[]
    # stemmer = english_stemmer
    # for word in words:
    #     b.append(stemmer.stem(word))

    return review_text.lower()

def read_data(path):
    pos_reviews = read_reviews(TRAIN_DIR + "/pos")
    neg_reviews = read_reviews(TRAIN_DIR + "/neg")
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

print "extract feature"

vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = None, ngram_range = ( 1, 4 ),
                              sublinear_tf = True )

vectorizer = vectorizer.fit(reviews_train + reviews_unsup)
feats_train = vectorizer.transform(reviews_train)
feats_test = vectorizer.transform(reviews_test)

print "perform feature selection"

fselect = SelectKBest(chi2 , k=70000)
feats_train = fselect.fit_transform(feats_train, labels_train)
feats_test = fselect.transform(feats_test)

print feats_train.shape, feats_test.shape
np.save("feats_train.npy", feats_train)
np.save("feats_test.npy", feats_test)
np.save("labels_train.npy", labels_train)
np.save("labels_test.npy", labels_test)

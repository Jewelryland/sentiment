import numpy as np
import cPickle as pickle
from preproc import preproc

print "Load models"
print 

with open("models/vectorizer_3.pkl", "rb") as fin:
    vectorizer = pickle.load(fin)

with open("models/fselect_3_30000.pkl", "rb") as fin:
    fselect = pickle.load(fin)

with open("models/clf_3_30000.pkl", "rb") as fin:
    clf = pickle.load(fin)

def classify(review, sent):
    review = review.strip()
    print sent, "review:"
    print review
    review = preproc(review)
    feat_ori = vectorizer.transform([review])
    feat = fselect.transform(feat_ori)
    print 
    pred = clf.predict(feat)
    if pred[0] == 1:
        ret = "positive"
    else:
        ret = "negative"
    print "Predict result:", ret
    print 


with open("pos_demo.txt") as fin:
    for review in fin:
        classify(review, "Positive")

with open("neg_demo.txt") as fin:
    for review in fin:
        classify(review, "Negative")

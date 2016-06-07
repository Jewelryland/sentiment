import re
import nltk
from bs4 import BeautifulSoup

def preproc(review, use_stopwords=False):
    review_text = BeautifulSoup(review, "lxml").get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    if use_stopwords:
        stops = set(nltk.stopwords.words("english"))
        words = [w for w in review_text.split() if not w in stops]
        return " ".join(words)

    return review_text.lower()


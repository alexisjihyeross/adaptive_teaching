import re
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import SyllableTokenizer
from nltk.corpus import cmudict
import nltk

VERB_CATEGORIES = {
    "+ed": lambda lemma, form: form == (lemma + "ed"),
    "+d": lambda lemma, form: form == (lemma + "d"),
    "+consonant+ed": lambda lemma, form: form == (lemma + lemma[-1] + "ed"),
    "y_to_ied": lambda lemma, form: lemma[-1] == "y" and form == lemma[:-1] + "ied",
}


def get_last_ngram(lemma, n):
    return lemma[-n:]


class LemmaNgramExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, ngram_range=(1, 3)):
        self.ngram_range = ngram_range
        self.SSP = SyllableTokenizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        ngrams = []
        for lemma in X:
            lemma_ngrams = []
            for i in range(*self.ngram_range):
                if i > len(lemma):
                    continue
                ngram = get_last_ngram(lemma, i)
                lemma_ngrams.append(ngram)

            ngrams.append(" ".join(lemma_ngrams))

        return np.array(ngrams)

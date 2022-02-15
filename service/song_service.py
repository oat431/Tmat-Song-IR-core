import pandas as pd
import string
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def get_all_song():
    data = pd.read_csv('./data/lyrics-data.csv')
    return data.head().to_dict()

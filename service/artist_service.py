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


def get_and_clean_data():
    data = pd.read_csv('./data/artists-data.csv')
    artist = data['Artist']
    artist = dict(artist)
    return artist

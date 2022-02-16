import pandas as pd
import json
import string
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from model.artists import Artists

data = pd.read_csv('./data/artists-data.csv')


def get_all_artists():
    artists = []
    for i in range(len(data)):
        artist = Artists(
            data.iloc[i].to_dict()["Artist"],
            data.iloc[i].to_dict()["Songs"],
            data.iloc[i].to_dict()["Popularity"],
            data.iloc[i].to_dict()["Link"],
            data.iloc[i].to_dict()["Genre"],
            data.iloc[i].to_dict()["Genres"]
        )
        artists.append(json.dumps(artist.get_artist()))
    return artists


def get_artists_by_name(name):
    query = name.lower()
    artists = []
    for i in range(len(data)):
        musician = data.iloc[i].to_dict()["Artist"].lower()
        if query in musician:
            artist = Artists(
                data.iloc[i].to_dict()["Artist"],
                data.iloc[i].to_dict()["Songs"],
                data.iloc[i].to_dict()["Popularity"],
                data.iloc[i].to_dict()["Link"],
                data.iloc[i].to_dict()["Genre"],
                data.iloc[i].to_dict()["Genres"]
            )
            artists.append(json.dumps(artist.get_artist()))

    if len(artists) == 0:
        return "not found"
    return artists


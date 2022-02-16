import json

import pandas as pd
import string
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

from model.Song import Songs
from model.bm25 import BM25
from model.querySong import QuerySongs

data = pd.read_csv('./data/lyrics-data.csv')
artist = pd.read_csv('./data/artists-data.csv')
artist_list = sorted([i.lower() for i in artist["Artist"]])
songs_list = sorted([i.lower() for i in data["SName"]])
data = data.drop_duplicates()
data = data[data.Idiom == "ENGLISH"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["Lyric"].astype('U'))
bm25 = BM25()
bm25.fit(data["Lyric"].astype('U'))

def get_all_song():
    return data.head().to_dict()


def get_song_by_name(name):
    songs = data[data.SName == name]
    found = []
    for i in range(len(songs)):
        song = Songs(
            songs.iloc[i].to_dict()["ALink"],
            songs.iloc[i].to_dict()["SName"],
            songs.iloc[i].to_dict()["Lyric"]
        )
        found.append(json.dumps(song.get_song()))
    return found


def clean_lyric(lyric):
    ps = PorterStemmer()
    s = word_tokenize(lyric)
    stopwords_set = set(stopwords.words())
    stop_dict = {s: 1 for s in stopwords_set}
    s = [w for w in s if w not in stop_dict]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s


def search_by_tf(query):
    pass
    # lyric = get_and_clean_lyric()
    # vectorizer = CountVectorizer(preprocessor=clean_lyric)
    # vectorizer.fit_transform(lyric)
    # result = vectorizer.transform([query])
    # return result


def serach_by_tf_idf(query):
    query_vec = vectorizer.transform([query])
    results = cosine_similarity(X,query_vec).reshape((-1,))
    return results.argsort()[-10:][::-1]


def search_by_bm25(query):
    result = bm25.transform(query,data[data.Idiom == "ENGLISH"]["Lyric"].astype('U'))
    return result.argsort()[-10:][::-1]


def search_song_by_lyric(query, score):
    lyric = query.lower()
    result = []
    rank = 1

    if lyric in artist_list:
        song = lyric.replace(" ", "-")
        song = "/" + song + "/"
        song_list = sorted([i for i in data[data.ALink == song]["SName"]])
        return song_list

    if lyric.lower() in songs_list:
        return get_song_by_name(query)

    if score == 'tf':
        return search_by_tf(query)

    if score == 'tf-idf':
        songs = serach_by_tf_idf(lyric)
        for i in songs:
            song = QuerySongs(
                rank,
                data.iloc[i].to_dict()["ALink"],
                data.iloc[i].to_dict()["SName"],
                query,
                clean_lyric(lyric)
            )
            rank += 1
            result.append(json.dumps(song.get_song()))
        return result
    if score == 'bm25':
        songs = search_by_bm25(lyric)
        for i in songs:
            song = QuerySongs(
                rank,
                data.iloc[i].to_dict()["ALink"],
                data.iloc[i].to_dict()["SName"],
                query,
                clean_lyric(lyric)
            )
            rank += 1
            result.append(json.dumps(song.get_song()))
        return result
    print('in correct method for seaching')
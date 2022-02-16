from flask import Flask, request, jsonify

from service.check_prime import check_prime
from service.artist_service import *
from service.song_service import *

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


# test api
@app.route('/check-prime/<int:num>')
def check_prime_api(num):
    if check_prime(num):
        txt = str(num) + " is Prime Number"
        return txt
    else:
        return str(num) + " is not Prime Number"


@app.route('/artists', methods=['GET'])
def get_artists():
    name = request.args.get("name")
    if name is None:
        return jsonify(get_all_artists())
    return jsonify(get_artists_by_name(name))


@app.route('/songs', methods=['GET'])
def get_songs():
    return get_all_song()


# search song
@app.route('/search-song', methods=['POST'])
def search_artist():
    return jsonify(search_song_by_lyric(request.json['query'], request.json['score']))


if __name__ == '__main__':
    app.run()

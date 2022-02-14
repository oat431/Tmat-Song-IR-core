from flask import Flask

from service.check_prime import check_prime
from service.artist_service import get_and_clean_data

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/check-prime/<int:num>')
def check_prime_api(num):
    if check_prime(num):
        txt = str(num) + " is Prime Number"
        return txt
    else:
        return str(num) + " is not Prime Number"


@app.route('/artists')
def get_artists():
    return get_and_clean_data()


if __name__ == '__main__':
    app.run()

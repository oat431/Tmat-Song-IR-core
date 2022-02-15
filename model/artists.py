def convert_to_list(genres) :
    if type(genres) == float:
        return []
    return genres.split('; ')


class Artists:
    def __init__(self, artists, song, popularity, link, genre, genres):
        self.artists = artists
        self.song = song
        self.popularity = popularity
        self.link = link
        self.genre = genre
        self.genres = genres

    def get_artist(self):
        return {
            "artists": self.artists,
            "song": self.song,
            "popularity": self.popularity,
            "link": self.link,
            "genre": self.genre,
            "genres": convert_to_list(self.genres)
        }

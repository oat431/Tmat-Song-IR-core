class Songs:
    def __init__(self, alink, sname, lyric):
        self.alink = alink
        self.sname = sname
        self.lyric = lyric

    def get_song(self):
        return {
            "name": self.sname,
            "lyric": self.lyric,
            "artist": self.alink,
        }

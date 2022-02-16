class QuerySongs:
    def __init__(self, rank, alink, sname, query_before, query_after):
        self.rank = rank
        self.alink = alink
        self.sname = sname
        self.query_before = query_before
        self.query_after = query_after

    def get_song(self):
        return {
            "rank": self.rank,
            "artist": self.alink,
            "song": self.sname,
            "queryBefore": self.query_before,
            "queryAfter": self.query_after,
        }
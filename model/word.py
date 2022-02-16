class RecommendWord :
    def __init__(self,correct_word,recommend_word):
        self.correct_word = correct_word
        self.recommend_word = recommend_word

    def get_word(self):
        return {
                "you mean this word": self.correct_word,
                "or this words": self.recommend_word
        }

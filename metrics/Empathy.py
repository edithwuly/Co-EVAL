from empath import Empath

empathy_categories = ["affection", "sympathy", "caring", "support", "sadness", "trust"]


def calculate_empathy(text):
    lexicon = Empath()
    analysis = lexicon.analyze(text, categories=empathy_categories)
    total_words = len(text.split())
    empathy_score = sum(analysis[category] / total_words for category in empathy_categories)
    return empathy_score
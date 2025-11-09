import re
import nltk
from nltk.corpus import cmudict

nltk.download('punkt')
nltk.download('cmudict')

d = cmudict.dict()


def count_syllables(word):
    word = word.lower()
    if word in d:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word]][0]
    else:
        return len(re.findall(r'[aeiouy]+', word.lower()))


def calculate_flesch_kincaid(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    total_sentences = len(sentences)
    total_words = len(words)
    total_syllables = sum(count_syllables(word) for word in words)

    reading_ease = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)

    return reading_ease
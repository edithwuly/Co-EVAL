import re


def count_syllables(word):
    word = word.lower()
    syllables = re.findall(r'[aeiouy]+', word)
    return len(syllables)


def is_complex_word(word):
    syllable_count = count_syllables(word)
    if syllable_count >= 3:
        if word.endswith(('es', 'ed')):
            if count_syllables(word[:-2]) < 3:
                return False
        return True
    return False


def calculate_gunning_fog_index(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    words = re.findall(r'\w+', text)
    num_words = len(words)
    num_sentences = len(sentences)
    num_complex_words = sum(1 for word in words if is_complex_word(word))

    if num_sentences == 0:
        return 0
    asl = num_words / num_sentences

    if num_words == 0:
        return 0
    phw = num_complex_words / num_words

    fog_index = 0.4 * (asl + phw * 100)

    return fog_index
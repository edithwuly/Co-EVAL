from nltk.translate.meteor_score import meteor_score


def calculate_meteor(source, reference, candidate):
    score = meteor_score([source.split(), reference.split()], candidate.split())

    return score
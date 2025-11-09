import math
from collections import Counter


def ngram_counts(sentence, n):
    return Counter([tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)])


def precision(reference, hypothesis, n):
    ref_counts = ngram_counts(reference, n)
    hyp_counts = ngram_counts(hypothesis, n)

    match_count = sum((hyp_counts & ref_counts).values())
    total_count = sum(hyp_counts.values())

    return match_count / total_count if total_count > 0 else 0


def recall(reference, hypothesis, n):
    ref_counts = ngram_counts(reference, n)
    hyp_counts = ngram_counts(hypothesis, n)

    match_count = sum((hyp_counts & ref_counts).values())
    total_count = sum(ref_counts.values())

    return match_count / total_count if total_count > 0 else 0


def calculate_gtm(reference, hypothesis, max_n=4):
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    precision_scores = [precision(ref_words, hyp_words, n) for n in range(1, max_n + 1)]
    recall_scores = [recall(ref_words, hyp_words, n) for n in range(1, max_n + 1)]

    geometric_means = [math.sqrt(p * r) for p, r in zip(precision_scores, recall_scores)]

    return sum(geometric_means) / max_n
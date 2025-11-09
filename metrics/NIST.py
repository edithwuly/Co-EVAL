from collections import Counter


def compute_ngram_precision(candidate, reference, n):
    def get_ngrams(text, n):
        return [tuple(text[i:i + n]) for i in range(len(text) - n + 1)]

    candidate_ngrams = get_ngrams(candidate, n)
    reference_ngrams = get_ngrams(reference, n)

    candidate_count = Counter(candidate_ngrams)
    reference_count = Counter(reference_ngrams)

    overlap_count = sum((candidate_count & reference_count).values())
    total_count = len(candidate_ngrams)

    precision = overlap_count / total_count if total_count > 0 else 0
    return precision


def calculate_nist(references, candidate, n=4):
    total_precision = 0
    for i in range(1, n + 1):
        precisions = [compute_ngram_precision(candidate, ref, i) for ref in references]
        avg_precision = sum(precisions) / len(precisions)
        total_precision += avg_precision

    return total_precision / n

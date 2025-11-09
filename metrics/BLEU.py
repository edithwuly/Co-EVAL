import math
import ast
from collections import Counter


def ngram_counts(sentence, n):
    return Counter([tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)])


def modified_precision(reference, hypothesis, n):
    ref_counts = ngram_counts(reference, n)
    hyp_counts = ngram_counts(hypothesis, n)
    clipped_counts = {ngram: min(count, ref_counts[ngram]) for ngram, count in hyp_counts.items()}

    return sum(clipped_counts.values()) / max(1, sum(hyp_counts.values()))


def brevity_penalty(reference, hypothesis):
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    if hyp_len > ref_len:
        return 1
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - ref_len / hyp_len)


def calculate_BLEU(reference, hypothesis, max_n=4):
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    precisions = [modified_precision(ref_words, hyp_words, n) for n in range(1, max_n + 1)]
    log_precisions = [math.log(p) for p in precisions if p > 0]

    if not log_precisions:
        geometric_mean = 0
    else:
        geometric_mean = math.exp(sum(log_precisions) / max_n)

    bp = brevity_penalty(ref_words, hyp_words)

    return bp * geometric_mean


def compute_exact_match(reference, hypothesis):
    return int(reference.strip() == hypothesis.strip())


def compute_ast_similarity(reference_code, hypothesis_code):
    try:
        reference_ast = ast.parse(reference_code)
        hypothesis_ast = ast.parse(hypothesis_code)
        return int(ast.dump(reference_ast) == ast.dump(hypothesis_ast))
    except:
        return 0


def compute_length_normalization(reference, hypothesis):
    return min(len(hypothesis) / len(reference), len(reference) / len(hypothesis))


def calculate_codeBLEU(reference_code, hypothesis_code):
    reference_tokens = reference_code.split()
    hypothesis_tokens = hypothesis_code.split()

    bleu_score = calculate_BLEU(reference_code, hypothesis_code)
    exact_match_score = compute_exact_match(reference_code, hypothesis_code)
    ast_similarity_score = compute_ast_similarity(reference_code, hypothesis_code)
    length_normalization_score = compute_length_normalization(reference_tokens, hypothesis_tokens)

    codebleu_score = (bleu_score + exact_match_score + ast_similarity_score + length_normalization_score) / 4.0
    return codebleu_score
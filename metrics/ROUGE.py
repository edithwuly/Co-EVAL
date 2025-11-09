from collections import Counter
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModel


def ngram_counts(sentence, n):
    return Counter([tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)])


def calculate_rouge_n(reference, hypothesis, n=2):
    ref_counts = ngram_counts(reference, n)
    hyp_counts = ngram_counts(hypothesis, n)

    overlap = sum((ref_counts & hyp_counts).values())
    total_ref_ngrams = sum(ref_counts.values())
    total_hyp_ngrams = sum(hyp_counts.values())

    precision = overlap / total_hyp_ngrams if total_hyp_ngrams > 0 else 0
    recall = overlap / total_ref_ngrams if total_ref_ngrams > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1_score


def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]


def calculate_rouge_l(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    lcs_length = lcs(ref_words, hyp_words)
    precision = lcs_length / len(hyp_words) if len(hyp_words) > 0 else 0
    recall = lcs_length / len(ref_words) if len(ref_words) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1_score


def weighted_lcs(ref_tokens: List[str], hyp_tokens: List[str], alpha: float = 1.0) -> Tuple[
    float, List[Tuple[int, int]]]:
    m, n = len(ref_tokens), len(hyp_tokens)
    L = np.zeros((m + 1, n + 1), dtype=float)

    backtrack = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                if i > 1 and j > 1 and ref_tokens[i - 2] == hyp_tokens[j - 2]:
                    prev_match_len = 1
                    k, l = i - 2, j - 2
                    while k >= 0 and l >= 0 and backtrack[k + 1, l + 1] == 3:
                        prev_match_len += 1
                        k -= 1
                        l -= 1

                    # Calculate new weight
                    new_match_len = prev_match_len + 1
                    new_weight = L[i - 1, j - 1] - (prev_match_len ** alpha) + (new_match_len ** alpha)

                    if new_weight > max(L[i - 1, j], L[i, j - 1]):
                        L[i, j] = new_weight
                        backtrack[i, j] = 3  # Diagonal (match)
                    elif L[i - 1, j] > L[i, j - 1]:
                        L[i, j] = L[i - 1, j]
                        backtrack[i, j] = 1  # Up
                    else:
                        L[i, j] = L[i, j - 1]
                        backtrack[i, j] = 2  # Left
                else:
                    # First match
                    new_weight = L[i - 1, j - 1] + 1.0 ** alpha

                    if new_weight > max(L[i - 1, j], L[i, j - 1]):
                        L[i, j] = new_weight
                        backtrack[i, j] = 3  # Diagonal (match)
                    elif L[i - 1, j] > L[i, j - 1]:
                        L[i, j] = L[i - 1, j]
                        backtrack[i, j] = 1  # Up
                    else:
                        L[i, j] = L[i, j - 1]
                        backtrack[i, j] = 2  # Left
            else:
                if L[i - 1, j] > L[i, j - 1]:
                    L[i, j] = L[i - 1, j]
                    backtrack[i, j] = 1  # Up
                else:
                    L[i, j] = L[i, j - 1]
                    backtrack[i, j] = 2  # Left

    # Backtrack to find the alignment
    alignment = []
    i, j = m, n
    while i > 0 and j > 0:
        if backtrack[i, j] == 3:  # Diagonal
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif backtrack[i, j] == 1:  # Up
            i -= 1
        else:  # Left
            j -= 1

    # Reverse the alignment to get it in correct order
    alignment.reverse()

    return L[m, n], alignment


def calculate_rouge_w(reference: str, hypothesis: str, alpha: float = 1.0):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    # Calculate weighted LCS
    wlcs_weight, _ = weighted_lcs(ref_tokens, hyp_tokens, alpha)

    # Calculate ROUGE-W Recall and Precision
    if len(ref_tokens) == 0:
        recall = 0.0
    else:
        recall = wlcs_weight / (len(ref_tokens) ** alpha)

    if len(hyp_tokens) == 0:
        precision = 0.0
    else:
        precision = wlcs_weight / (len(hyp_tokens) ** alpha)

    # Calculate F1 score
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    return f1_score


def get_bert_model_and_tokenizer(model_name="bert-base-uncased"):
    """Load BERT model and tokenizer from HuggingFace"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer


def get_token_embeddings(text: str, model, tokenizer):
    """Get BERT embeddings for each token in the text"""
    # Add special tokens and convert to tensor
    encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=True)

    # Get token IDs and attention mask
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    # Get token map to handle subword tokenization
    token_map = []
    tokens = text.split()
    for i, token in enumerate(tokens):
        subtokens = tokenizer.tokenize(token)
        token_map.extend([i] * len(subtokens))

    # Account for [CLS] token
    token_map = [-1] + token_map

    # Truncate to match model's input
    if len(token_map) > input_ids.size(1):
        token_map = token_map[:input_ids.size(1)]

    # Pad token map if necessary
    while len(token_map) < input_ids.size(1):
        token_map.append(-1)

    # Get embeddings
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        # Use last hidden state
        embeddings = outputs.last_hidden_state[0].numpy()

    # Group subword embeddings by original token
    token_embeddings = {}
    for i, token in enumerate(tokens):
        # Find all positions where this token appears
        positions = [j for j, idx in enumerate(token_map) if idx == i]
        if positions:
            # Average the embeddings for all subwords of this token
            token_emb = embeddings[positions].mean(axis=0)
            token_embeddings[token.lower()] = token_emb

    return token_embeddings


def token_similarity(token1: str, token2: str, token_embeddings_1: Dict[str, np.ndarray],
                     token_embeddings_2: Dict[str, np.ndarray], eps: float = 1e-9) -> float:
    """Calculate cosine similarity between two tokens using BERT embeddings"""
    if token1 not in token_embeddings_1 or token2 not in token_embeddings_2:
        return 0.0

    vec1 = token_embeddings_1[token1]
    vec2 = token_embeddings_2[token2]

    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return dot_product / (norm1 * norm2 + eps)


def find_best_similarity(token: str, other_tokens: List[str],
                         token_embeddings_1: Dict[str, np.ndarray],
                         token_embeddings_2: Dict[str, np.ndarray]) -> float:
    """Find the highest similarity between a token and a list of other tokens"""
    if not other_tokens:
        return 0.0

    similarities = [token_similarity(token, other, token_embeddings_1, token_embeddings_2)
                    for other in other_tokens]
    return max(similarities)


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def ngram_similarity(ngram1: Tuple[str, ...], ngram2: Tuple[str, ...],
                     token_embeddings_1: Dict[str, np.ndarray],
                     token_embeddings_2: Dict[str, np.ndarray]) -> float:
    """Calculate average similarity between two n-grams"""
    if len(ngram1) != len(ngram2):
        return 0.0

    position_sims = [token_similarity(t1, t2, token_embeddings_1, token_embeddings_2)
                     for t1, t2 in zip(ngram1, ngram2)]
    return sum(position_sims) / len(ngram1) if position_sims else 0.0


def calculate_rouge_we_n(reference: str, hypothesis: str, model, tokenizer,
                    n: int = 2, threshold: float = 0.0, eps: float = 1e-9) -> float:
    ref_tokens = [t.lower() for t in reference.split()]
    hyp_tokens = [t.lower() for t in hypothesis.split()]

    if len(ref_tokens) < n or len(hyp_tokens) < n:
        return 0.0

    # Get contextual embeddings
    ref_embeddings = get_token_embeddings(reference, model, tokenizer)
    hyp_embeddings = get_token_embeddings(hypothesis, model, tokenizer)

    ref_ngrams = get_ngrams(ref_tokens, n)
    hyp_ngrams = get_ngrams(hyp_tokens, n)

    # Calculate n-gram similarities
    hyp_to_ref_scores = []
    for hyp_gram in hyp_ngrams:
        scores = []
        for ref_gram in ref_ngrams:
            avg_sim = ngram_similarity(hyp_gram, ref_gram, hyp_embeddings, ref_embeddings)
            scores.append(avg_sim)

        if scores and max(scores) > threshold:
            hyp_to_ref_scores.append(max(scores))

    ref_to_hyp_scores = []
    for ref_gram in ref_ngrams:
        scores = []
        for hyp_gram in hyp_ngrams:
            avg_sim = ngram_similarity(ref_gram, hyp_gram, ref_embeddings, hyp_embeddings)
            scores.append(avg_sim)

        if scores and max(scores) > threshold:
            ref_to_hyp_scores.append(max(scores))

    # Calculate precision and recall
    precision = sum(hyp_to_ref_scores) / len(hyp_ngrams) if hyp_ngrams else 0.0
    recall = sum(ref_to_hyp_scores) / len(ref_ngrams) if ref_ngrams else 0.0

    # Calculate F1 score
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return f1
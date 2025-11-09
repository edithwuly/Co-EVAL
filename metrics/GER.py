import re
from typing import List, Tuple, Dict, Set, Optional
import numpy as np


def preprocess_text(text: str) -> List[str]:
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Split into tokens
    tokens = text.split()
    return tokens


def get_error_spans(text: str, corrected_text: str,
                    annotations: Optional[List[Tuple[int, int, str]]] = None) -> List[Tuple[int, int]]:
    if annotations:
        # If explicit annotations are provided, use them
        return [(start, end) for start, end, _ in annotations]
    else:
        orig_tokens = preprocess_text(text)
        corr_tokens = preprocess_text(corrected_text)

        error_spans = []
        i, j = 0, 0
        start_error = None

        while i < len(orig_tokens) and j < len(corr_tokens):
            if orig_tokens[i] != corr_tokens[j]:
                if start_error is None:
                    start_error = i
                i += 1
            else:
                if start_error is not None:
                    error_spans.append((start_error, i))
                    start_error = None
                i += 1
                j += 1

        # Check if there's an ongoing error at the end
        if start_error is not None:
            error_spans.append((start_error, len(orig_tokens)))

        return error_spans


def calculate_ger(corrected_text: str, text: str,
                  annotations: Optional[List[Tuple[int, int, str]]] = None) -> float:
    tokens = preprocess_text(text)
    error_spans = get_error_spans(text, corrected_text, annotations)

    # Count the number of tokens in error spans
    error_token_count = 0
    for start, end in error_spans:
        error_token_count += (end - start)

    if len(tokens) == 0:
        return 0.0

    # GER is the ratio of tokens with errors to total tokens
    ger = error_token_count / len(tokens)
    return ger
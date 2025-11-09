def calculate_distinct_n(texts, n=4):
    ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams += [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    # print(ngrams)
    total_ngrams = len(ngrams)
    unique_ngrams = len(set(ngrams))
    # print(unique_ngrams)

    return unique_ngrams / total_ngrams if total_ngrams > 0 else 0
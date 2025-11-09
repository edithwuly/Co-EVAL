def calculate_length_ratio(article, summary):
    summary_length = len(summary.split())
    article_length = len(article.split())

    length_ratio = summary_length / article_length

    return length_ratio
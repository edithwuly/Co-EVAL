import argparse

from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr, kendalltau
import json
import re


def calculate_correlation(pred_score, human_score, result):
    assert len(pred_score) == len(human_score)

    if (len(result) == 0):
        result = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
    result['pearson'] += pearsonr(pred_score, human_score)[0]
    result['spearman'] += spearmanr(pred_score, human_score)[0]
    result['kendalltau'] += kendalltau(pred_score, human_score)[0]

    return result


def print_correlations(result, n):
    table = PrettyTable(['Pearson', 'Spearman', 'Kendall'])
    if (n == 0):
        n = 1
    table.add_row(
        [round(result['pearson'] / n, 3), round(result['spearman'] / n, 3), round(result['kendalltau'] / n, 3)])
    print(table)


def parse_output(output):
    matched = re.search("^ ?([\d\.]+)", output)
    if (matched):
        try:
            score = float(matched.group(1))
        except:
            score = 0
    else:
        score = 0
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_file", type=str, required=True, help="Path to the score file.")
    args = parser.parse_args()

    res = []
    with open(args.score_file) as f:
        for line in f.readlines():
            res.append(json.loads(line))

    scores, human_scores = {}, {}

    print("Calculating correlation for My-Eval")
    for idx, item in enumerate(res):
        doc_id = item["doc_id"]
        system_id = item["system_id"]
        if (doc_id not in scores):
            scores[doc_id] = []
            human_scores[doc_id] = []

        scores[doc_id].append(item["score"])
        human_scores[doc_id].append(item["human"])

    results_naive = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
    d_ctr = 0
    for doc_id in scores.keys():
        scores_doc = scores[doc_id]
        human_scores_doc = human_scores[doc_id]
        if ((len(set(human_scores_doc)) <= 1) or (len(set(scores_doc)) <= 1)) :
            continue
        results_naive = calculate_correlation(scores_doc, human_scores_doc, results_naive)
        d_ctr += 1

    print_correlations(results_naive, n=d_ctr)
import argparse

from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr, kendalltau
import json
import re


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    args = parser.parse_args()

    models = ["llama-3-8b", "llama-3-70b", "qwen-2-7b", "qwen-2-72b", "gemma-2-9b", "gemma-2-27b"]

    res = {}
    for model in models:
        with open(f"experiment2/results/{args.dataset}_{args.type}_{model}.jsonl") as f:
            for line in f.readlines():
                line = json.loads(line)
                doc_id = str(line["doc_id"])
                system_id = line["system_id"]
                score = line["score"]
                if doc_id not in res.keys():
                    res[doc_id] = {}
                if model not in res[doc_id].keys():
                    res[doc_id][model] = []
                res[doc_id][model].append((system_id, score))

    for system_ids in res.values():
        for key, scores in system_ids.items():
            scores.sort(key=lambda x: x[1], reverse=True)
            ranked_scores = []
            current_rank = 1
            for idx, score in enumerate(scores):
                if idx > 0 and score[1] != scores[idx - 1][1]:
                    current_rank = idx + 1
                ranked_scores.append((score[0], current_rank))

            system_ids[key] = ranked_scores


    self_rank = {}
    other_rank = {}
    for doc_id, eval_models in res.items():
        if doc_id not in self_rank.keys():
            self_rank[doc_id] = {}
        if doc_id not in other_rank.keys():
            other_rank[doc_id] = {}
        for eval_model, scores in eval_models.items():
            for score in scores:
                system_id = score[0]
                if eval_model == system_id:
                    self_rank[doc_id][system_id] = score[1]
        for eval_model, scores in eval_models.items():
            for score in scores:
                system_id = score[0]
                if eval_model != system_id:
                    if system_id not in other_rank[doc_id].keys():
                        other_rank[doc_id][system_id] = []
                    other_rank[doc_id][system_id].append(score[1])

    bias = {}
    for model in models:
        bias[model] = 0

    for doc_id, ranks in self_rank.items():
        for model, rank in ranks.items():
            average_rank = sum(other_rank[doc_id][model]) / len(other_rank[doc_id][model])
            if rank < average_rank:
                bias[model] += abs(average_rank - rank)

    table = PrettyTable(models)
    table.add_row([round(bias[i]/len(res.keys()), 4) for i in models])
    print(table)
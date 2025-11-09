import argparse
import json
import os.path
import re

import torch
from openai import OpenAI
from tqdm import tqdm

import prompts
from generate_criteria import CriteriaPlanner
from retrieve_metrics import MetricLibrary


class Evaluator:
    def __init__(self, model, api_key, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, content):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model=self.model,
            temperature=0.7,
            stop=["<|eot_id|>"]
        )
        return chat_completion.choices[0].message.content

    def parse_batch_score(self, content, batch_size):
        matches = re.findall(r'Sample\s*(\d+):\s*([\d.]+)', content)
        scores = [0] * batch_size

        for match in matches:
            index = int(match[0]) - 1
            score = float(match[1])
            scores[index] = score
        return scores

    def evaluate(self, task_name, task_description, data, batch_size, criteria, output_dir):
        batch_data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        count = 5
        with open(os.path.join(output_dir, f"{task_name}_co-eval_{self.model}.jsonl"), "w") as f:
            for batch in tqdm(batch_data):
                total_score = [0 for _ in range(batch_size)]
                for item in criteria:
                    for metric in item["metrics"]:
                        scores = [0 for _ in range(batch_size)]
                        metric_implementation = metric["implementation"]
                        for idx, response in enumerate(batch):
                            scores[idx] = metric_implementation(response["source"], response["system_output"], device)
                        metric["scores"] = scores
                    print(item)
                    prompt = prompts.evaluate(task_description, item, batch, batch_size)
                    print(prompt)
                    scores = [[] for _ in range(batch_size)]
                    for _ in range(count):
                        try:
                            res = self.parse_batch_score(self.chat(prompt), batch_size)
                            for idx, value in enumerate(res):
                                scores[idx].append(value)
                        except Exception:
                            for i in range(batch_size):
                                scores[i].append(-1)
                    for idx, value in enumerate(scores):
                        score = sum(scores[idx]) / len(scores[idx])
                        total_score[idx] += score
                    print(f"co-eval for {batch[0]['doc_id']} in {item['name']}: {scores}")
                print(f"co-eval for {batch[0]['doc_id']}: {total_score}")
                for idx, instance in enumerate(batch):
                    f.write(json.dumps({"doc_id": instance["doc_id"],
                               "system_id": instance["system_id"],
                               "score": total_score[idx]}))
                    f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="conala", help="Description for the task to be evaluated.")
    parser.add_argument("--task_description", type=str, default="Generate executable Python code for the given requirement.", help="Description for the task to be evaluated.")
    parser.add_argument("--embedding_model_path", type=str, default="../../models/bert-base-uncased", help="Path to embedding model.")
    parser.add_argument("--response_path", type=str, default="data/conala_generated.json", help="Path to generated responses.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of generated responses.")
    parser.add_argument("--output_dir", type=str, default="result", help="Directory to save output")
    parser.add_argument("--planner_model", type=str, default="Llama-3-8B-Instruct", help="Model as criteria planner")
    parser.add_argument("--evaluator_model", type=str, default="gpt-4o", help="Model as prompt-based evaluator.")

    args = parser.parse_args()

    criteria_planner = CriteriaPlanner(args.planner_model, "your planner api key", "your planner base url")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_library = MetricLibrary(args.embedding_model_path, device)

    task_description = args.task_description
    criteria = criteria_planner.generate_criteria(task_description)
    print(criteria)

    for item in criteria:
        criteria_description = item["description"]
        machine_metrics = metric_library.retrieve_metrics(criteria_description)
        for machine_metric in machine_metrics:
            refined_description = criteria_planner.refine_metric(item, machine_metric)
            machine_metric["description"] = refined_description
        item["metrics"] = machine_metrics

    print(criteria)

    evaluator = Evaluator(args.evaluator_model, "your evaluator api key", "your evaluator base url")
    data = json.load(open(args.response_path))
    evaluator.evaluate(args.task_name, task_description, data, args.batch_size, criteria, args.output_dir)
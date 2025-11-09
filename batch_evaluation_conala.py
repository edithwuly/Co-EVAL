import argparse
import json
import re

from openai import OpenAI
from tqdm import tqdm


def batch_evaluation_without_machine_metric(batch_content):
    content_str = ""
    for idx, content in enumerate(batch_content):
        content_str += f"Sample{idx + 1}:\nSystem Response: {content['system_output'].strip()}\n"

    return f'''You will be given a batch of 8 samples. Each sample contains a generated code for given requirement.

Your task is to assign a float score to the response on one metric.

You should carefully horizontally compare the given samples in order to assign a suitable float score to each sample.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Overall (floating point numbers within the interval [1,5]): What is your overall impression of the quality of the generated code?
- A float score near 1 (very poor): The generated code is of very low quality. It contains significant errors or does not run at all, lacks any meaningful structure, and does not meet the requirements in any substantial way. The code might be difficult or impossible to salvage for further use.
- A float score near 2 (poor): The code runs but is largely incorrect or ineffective. There are numerous logical errors or missing functionality, and it does not align well with the provided requirements. The code may also suffer from poor readability or lack of proper structure, making it difficult to understand or maintain.
- A float score near 3 (neutral): The code is functional but unremarkable. It may have some errors or areas for improvement but generally follows the basic requirements and runs with acceptable results. The code is neither highly readable nor efficient, but it’s not overly difficult to understand or extend.
- A float score near 4 (good): The generated code is of good quality, meeting most of the requirements with only minor issues. It runs correctly for the majority of test cases and is fairly easy to read and maintain. The code could be improved, but any changes would be enhancements rather than necessary fixes.
- A float score near 5 (excellent): The code is of very high quality, demonstrating strong adherence to all requirements. It is free from significant errors, highly readable, well-structured, efficient, and maintainable. The code is clear, concise, and easy to understand, with well-considered logic and style. There are no significant flaws or areas for improvement.


Generated code and given requirement:

Source: {batch_content[0]['source'].strip()}
{content_str}

Evaluation Form (Answer by starting with "I will do my best to provide individual analysis for each sample. Analysis:" to analyze the given samples regarding the evaluation criteria as concise as possible (Attention: Don't give your scores during this step). After analysing all the samples, please give all the float scores in order following the template "Float Scores: [Sample1:score of Sample1,Sample2:score of Sample2,Sample3:score of Sample3,Sample4:score of Sample4,Sample5:score of Sample5,Sample6:score of Sample6,Sample7:score of Sample7,Sample8:score of Sample8]".
- Overall:'''


def parse_batch_score(content, batch_size):
    print(content)
    matches = re.findall(r'\**\**Sample\s*(\d+)\**\**:\**\**\s*([\d.]+)', content)
    scores = [0] * batch_size

    for match in matches:
        index = int(match[0]) - 1
        score = float(match[1])
        scores[index] = score
    return scores


client = OpenAI(api_key="your api key", base_url="your base url")


def chat(content, model):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        model=model,
        stop="<|eot_id|>",
        temperature=0.7,
    )
    return chat_completion.choices[0].message.content


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_path", type=str, default="../data/conala_generated.json", help="Path to generated responses.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model as evaluator.")
    args = parser.parse_args()

    data = json.load(open(args.response_path))
    model = args.model
    with open(f"result/conala_batch_{model}.jsonl", "w") as f:
        batch_size = 8
        batch_data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        count = 5
        for batch in tqdm(batch_data):
            prompt = batch_evaluation_without_machine_metric(batch)
            scores = [[] for _ in range(batch_size)]
            for _ in range(count):
                try:
                    res = parse_batch_score(chat(prompt, model), batch_size)
                    for idx, value in enumerate(res):
                        scores[idx].append(value)
                except Exception:
                    for i in range(batch_size):
                        scores[i].append(-1)
            print(f"batch eval for {batch[0]['doc_id']}: {scores}")
            for idx, value in enumerate(scores):
                score = sum(scores[idx]) / len(scores[idx])
                f.write(json.dumps({"doc_id": batch[idx]["doc_id"],
                                    "system_id": batch[idx]["system_id"],
                                    "score": score}))
                f.write("\n")
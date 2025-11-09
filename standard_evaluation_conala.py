import argparse
import json
import re

from openai import OpenAI
from tqdm import tqdm


def evaluation_without_machine_metric(content):
    return f'''You will be given a sample, containing a generated code for given requirement.

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

Source: {content['source'].strip()}
System Response: {content['system_output'].strip()}

Evaluation Form (scores ONLY):
- Overall:'''


def parse_score(content):
    match = re.search(r'([\d.]+)', content)
    if match:
        return float(match.group(1))
    else:
        return 0


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
    with open(f"result/conala_standard_{model}.jsonl", "w") as f:
        count = 5
        for instance in tqdm(data):
            prompt = evaluation_without_machine_metric(instance)
            scores = []
            for _ in range(count):
                try:
                    res = parse_score(chat(prompt, model))
                    scores.append(res)
                except Exception:
                    scores.append(-1)
            score = sum(scores) / len(scores)
            f.write(json.dumps({"doc_id": instance["doc_id"],
                                "system_id": instance["system_id"],
                                "score": score}))
            f.write("\n")

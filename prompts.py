def generate_criteria(task_description):
    return f'''Please provide the evaluation criteria for this task, including the weight of each criterion. The total score should be 10 points.
    
Task: {task_description}'''


def generate_criteria_preparation(task_description):
    return f'''Task: {task_description}.

Instruction: Please provide the evaluation criteria for this task, including the weight of each criterion. The total score should be 10 points, with no more than 5 criteria in total. Present the information in the following format:
No. Criterion Name (Weight in points) - Description of what this criterion evaluates. Provide clear guidance on how this aspect of the response will be assessed.

An Example:
1. Efficiency (2 points): Is the generated code optimized in terms of time and space complexity?
- A float score near 0 (no) means the code is inefficient and has significant room for optimization.
- A float score near 1 (somewhat) means the code has a moderate level of efficiency but could be improved.
- A float score near 2 (yes) means the code is highly optimized in both time and space complexity.

Return the complete list. Note: Efficiency is included as an example and is not required to be part of the final list.'''


def refine_metric_description(criteria, machine_metric):
    return f'''Please provide a detailed metric description that clearly explains how the metric reflects and aligns with the corresponding criterion.
Criteria: {criteria["name"]} - {criteria["description"]}
Machine Metric: {machine_metric["name"]} - {machine_metric["description"]}'''


def generate_metric_description_preparation(criteria):
    return f'''Instruction: First, generate the most suitable machine metric for the given criterion with metric description. Then, provide a detailed metric description that clearly explains how the metric reflects and aligns with the corresponding criterion.

Example:
Criteria: Coherence – Measures how logically the summary flows, ensuring clarity and consistency in the ideas presented.
Machine Metric: BERTScore – Evaluates the semantic similarity between two pieces of text.
Detailed Machine Metric: BERTScore – Evaluates the semantic similarity between two pieces of text. A higher BERTScore reflects a greater degree of coherence, indicating that the summary aligns more closely with the logical flow and meaning of the original content.

Criteria: {criteria["name"]} - {criteria["description"]}'''


def refine_metric_description_preparation(criteria, machine_metric):
    return f'''Instruction: Refine the given metric description to clearly explain how the metric reflects and aligns with the corresponding criterion.

Example:
Criteria: Coherence – Measures how logically the summary flows, ensuring clarity and consistency in the ideas presented.
Machine Metric: BERTScore – Evaluates the semantic similarity between two pieces of text.
Detailed Machine Metric: BERTScore – Evaluates the semantic similarity between two pieces of text. A higher BERTScore reflects a greater degree of coherence, indicating that the summary aligns more closely with the logical flow and meaning of the original content.

Criteria: {criteria["name"]} - {criteria["description"]}
Machine Metric: {machine_metric["name"]} - {machine_metric["description"]}'''


def evaluate(task_description, criteria, batch_content, batch_size):
    criteria_str = f"{criteria['name']} (floating point numbers within the interval [0,{criteria['weight']}]): {criteria['description']}\n"
    for item in criteria['details']:
        criteria_str += f"- {item}\n"

    content_str = ""
    answer_format = []
    for content_idx, content in enumerate(batch_content):
        content_str += f"Sample{content_idx + 1}:\nSystem Response: {content['system_output'].strip()}\n"
        for metric in criteria['metrics']:
            content_str += f"{metric['name']} - {metric['description']}\nScore: {metric['scores'][content_idx]}\n"
        answer_format.append(f"Sample{content_idx + 1}:score of Sample{content_idx + 1}")

    return f'''You will be given a batch of {batch_size} samples for the task: {task_description}

Your task is to assign a float score to the response on one metric.

You should carefully horizontally compare the given samples in order to assign a suitable float score to each sample.

You can refer to the machine metric scores of each sample if you are not confidence.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

{criteria_str}

Given Content and potentially useful Machine Metric Score:
Source: {batch_content[0]['source'].strip()}
{content_str}

Evaluation Form (Answer by starting with "Analysis:" to analyze the given samples regarding the evaluation criteria and offer insights derived from the machine metric scores as concise as possible (Attention: Don't give your scores during this step). After analysing all the samples, please give all the float scores in order following the template "Float Scores: [{','.join(answer_format)}]".
- {criteria['name']}:'''
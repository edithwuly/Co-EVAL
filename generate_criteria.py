import re

from openai import OpenAI
import prompts


class CriteriaPlanner:
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
            stop=["<|eot_id|>"]
        )
        return chat_completion.choices[0].message.content

    def parse_criteria(self, response):
        pattern = re.compile(r"(\d+)\.\s+(.*?)\s+\((\d+)\s+points?\):\s+(.*?)\n(-.*?)\n\n", re.DOTALL)

        criteria = []
        for match in pattern.finditer(response):
            name = match.group(2).strip()
            weight = int(match.group(3).strip())
            description = match.group(4).strip()
            details = [detail.strip("- ") for detail in match.group(5).split('\n- ') if detail]

            criteria.append({
                "name": name,
                "weight": weight,
                "description": description,
                "details": details
            })
        return criteria

    def parse_refined_description(self, response):
        pattern = r"Detailed Machine Metric: [\w\s-]+ - (.+)"
        match = re.search(pattern, response)
        detailed_description = match.group(1).strip()
        return detailed_description

    def generate_criteria(self, task_description):
        prompt = prompts.generate_criteria(task_description)
        response = self.chat(prompt)
        return self.parse_criteria(response)

    def refine_metric(self, criteria, machine_metric):
        prompt = prompts.refine_metric_description(criteria, machine_metric)
        response = self.chat(prompt)
        return self.parse_refined_description(response)
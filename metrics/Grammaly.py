import json


def calculate_grammaly(doc_id, system_id):
    with open("result/grammaly.jsonl") as f:
        for line in f.readlines():
            line = json.loads(line)
            if line["doc_id"] == doc_id and line["system_id"] == system_id:
                return line["score"]
import json


def calculate_sonar_maintainability(doc_id, system_id):
    with open("result/sonar_maintainability.jsonl") as f:
        for line in f.readlines():
            line = json.loads(line)
            if line["doc_id"] == doc_id and line["system_id"] == system_id:
                return line["score"]


def calculate_sonar_reliability(doc_id, system_id):
    with open("result/sonar_reliability.jsonl") as f:
        for line in f.readlines():
            line = json.loads(line)
            if line["doc_id"] == doc_id and line["system_id"] == system_id:
                return line["score"]


def calculate_sonar_coverage(doc_id, system_id):
    with open("result/sonar_coverage.jsonl") as f:
        for line in f.readlines():
            line = json.loads(line)
            if line["doc_id"] == doc_id and line["system_id"] == system_id:
                return line["score"]


def calculate_sonar_duplication(doc_id, system_id):
    with open("result/sonar_duplication.jsonl") as f:
        for line in f.readlines():
            line = json.loads(line)
            if line["doc_id"] == doc_id and line["system_id"] == system_id:
                return line["score"]


def calculate_sonar_security(doc_id, system_id):
    with open("result/sonar_security.jsonl") as f:
        for line in f.readlines():
            line = json.loads(line)
            if line["doc_id"] == doc_id and line["system_id"] == system_id:
                return line["score"]
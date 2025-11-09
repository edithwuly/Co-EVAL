import torch
from transformers import AutoTokenizer, AutoModel


def calculate_completeness(response, device):
    tokenizer = AutoTokenizer.from_pretrained("mathbert")
    model = AutoModel.from_pretrained("mathbert").to(device)

    steps = response.split("\n")

    def get_step_embeddings(steps):
        embeddings = []
        for step in steps:
            inputs = tokenizer(step, return_tensors="pt", truncation=True, padding=True, max_length=512)

            with torch.no_grad():
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

            embeddings.append(cls_embedding)

        return embeddings

    step_embeddings = get_step_embeddings(steps)

    similarity_scores = []
    for i in range(1, len(step_embeddings)):
        cosine_sim = torch.nn.functional.cosine_similarity(step_embeddings[i], step_embeddings[i - 1])
        similarity_scores.append(cosine_sim.item())

    if len(similarity_scores) == 0:
        score = 0
    else:
        score = sum(similarity_scores) / len(similarity_scores)

    return score
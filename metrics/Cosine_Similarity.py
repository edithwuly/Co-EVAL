import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def get_embedding(input, model, device):
    with torch.no_grad():
        outputs = model(**input)

    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().to(device)

    return cls_embedding


def calculate_cosine_similarity(sentence1, sentence2, device):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    tokens1 = tokenizer(sentence1, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
    tokens2 = tokenizer(sentence2, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
    tokens1 = {k: v.to(device) for k, v in tokens1.items()}
    tokens2 = {k: v.to(device) for k, v in tokens2.items()}
    embedding1 = get_embedding(tokens1, model, device)
    embedding2 = get_embedding(tokens2, model, device)

    embedding1 = embedding1.cpu().numpy()
    embedding2 = embedding2.cpu().numpy()

    cosine_sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return float(cosine_sim)

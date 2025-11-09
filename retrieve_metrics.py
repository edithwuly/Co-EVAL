import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from metrics import metric_info


class MetricLibrary:
    def __init__(self, embedding_model_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
        self.model = AutoModel.from_pretrained(embedding_model_path).to(self.device)
        self.metric_info = metric_info.info
        self.metric_embeddings = []

        self.embedding_dim = len(self.generate_embedding(list(self.metric_info.values())[0]["description"]))
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        self._build_index()

    def generate_embedding(self, text):
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(
            self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embedding.flatten()

    def _build_index(self):
        for name, info in self.metric_info.items():
            embedding = self.generate_embedding(info["description"])
            faiss.normalize_L2(embedding.reshape(1, -1))
            self.index.add(np.array([embedding]))
            self.metric_embeddings.append((name, info["description"]))

    def retrieve_metrics(self, criteria_description, top_k=3, similarity_threshold=0.7):
        criteria_embedding = self.generate_embedding(criteria_description)
        faiss.normalize_L2(criteria_embedding.reshape(1, -1))

        distances, indices = self.index.search(np.array([criteria_embedding]), top_k)

        similar_metrics = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity_score = dist
            if similarity_score >= similarity_threshold:
                metric_name, metric_description = self.metric_embeddings[idx]
                similar_metrics.append({
                    "name": metric_name,
                    "description": metric_description,
                    "implementation": self.metric_info[metric_name]["implementation"],
                    "similarity": similarity_score
                })

        return similar_metrics
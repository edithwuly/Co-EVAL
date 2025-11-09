import torch
from transformers import AutoTokenizer, AutoModel


def calculate_semantic(response, device):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to(device)
    inputs = tokenizer(response.strip(), truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    score = torch.max(probs).item()

    return score
import torch
from transformers import AutoTokenizer, AutoModel


def calculate_factcc(sentence1, sentence2, device):
    tokenizer = AutoTokenizer.from_pretrained("FactCC")
    model = AutoModel.from_pretrained("FactCC").to(device)
    inputs = tokenizer.encode_plus(sentence1, sentence2, add_special_tokens=True, return_tensors="pt",truncation=True, padding="max_length", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred = logits.argmax(dim=1)
    return int(pred.item() == "0")

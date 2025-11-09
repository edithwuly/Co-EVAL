import torch
from transformers import AutoTokenizer, AutoModel


def calculate_bartscore(reference, candidate, device):
    model = AutoModel.from_pretrained("bart-large-cnn").to(device)
    tokenizer = AutoTokenizer.from_pretrained("bart-large-cnn")
    sentence1 = tokenizer(reference, truncation=True, padding='max_length', max_length=512,
                          return_tensors='pt').to("cuda")
    sentence2 = tokenizer(candidate, truncation=True, padding='max_length', max_length=512,
                          return_tensors='pt').to("cuda")
    with torch.no_grad():
        outputs = model(**sentence1, labels=sentence2.input_ids)

    log_likelihood = -outputs.loss.item()

    return log_likelihood
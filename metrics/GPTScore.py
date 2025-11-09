import torch
from transformers import AutoTokenizer, AutoModel

def calculate_gpt_score(response, device):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2").to(device)
    inputs = tokenizer(response, truncation=True, max_length=512, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']

    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)

    log_likelihood = -outputs.loss.item()

    return log_likelihood
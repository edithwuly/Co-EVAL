import torch
from transformers import AutoTokenizer, AutoModel


def calculate_perplexity(content, device):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2").to(device)
    content = tokenizer(content.strip(), truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        input_ids = content['input_ids'].to(device)

        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = torch.exp(torch.tensor(loss))

    return perplexity.item()
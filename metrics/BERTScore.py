import torch
from transformers import AutoTokenizer, AutoModel


def calculate_bertscore(reference, candidate, device):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    ref_tokens = tokenizer(reference, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
    cand_tokens = tokenizer(candidate, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
    ref_tokens = {k: v.to(device) for k, v in ref_tokens.items()}
    cand_tokens = {k: v.to(device) for k, v in cand_tokens.items()}

    with torch.no_grad():
        ref_embeddings = model(**ref_tokens).last_hidden_state
        cand_embeddings = model(**cand_tokens).last_hidden_state

    ref_embeddings = ref_embeddings.squeeze(0)
    cand_embeddings = cand_embeddings.squeeze(0)
    cosine_sim = torch.nn.functional.cosine_similarity(ref_embeddings.unsqueeze(1), cand_embeddings.unsqueeze(0), dim=-1)

    ref_to_cand = cosine_sim.max(dim=1).values
    cand_to_ref = cosine_sim.max(dim=0).values

    precision = ref_to_cand.mean().item()
    recall = cand_to_ref.mean().item()

    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return f1_score


def calculate_codebertscore(reference, candidate, device):
    tokenizer = AutoTokenizer.from_pretrained("codebert-base")
    model = AutoModel.from_pretrained("codebert-base").to(device)
    ref_tokens = tokenizer(reference, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
    cand_tokens = tokenizer(candidate, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
    ref_tokens = {k: v.to(device) for k, v in ref_tokens.items()}
    cand_tokens = {k: v.to(device) for k, v in cand_tokens.items()}

    with torch.no_grad():
        ref_embeddings = model(**ref_tokens).last_hidden_state
        cand_embeddings = model(**cand_tokens).last_hidden_state

    ref_embeddings = ref_embeddings.squeeze(0)
    cand_embeddings = cand_embeddings.squeeze(0)
    cosine_sim = torch.nn.functional.cosine_similarity(ref_embeddings.unsqueeze(1), cand_embeddings.unsqueeze(0), dim=-1)

    ref_to_cand = cosine_sim.max(dim=1).values
    cand_to_ref = cosine_sim.max(dim=0).values

    precision = ref_to_cand.mean().item()
    recall = cand_to_ref.mean().item()

    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return f1_score


def calculate_finbertscore(reference, candidate, device):
    tokenizer = AutoTokenizer.from_pretrained("finbert")
    model = AutoModel.from_pretrained("finbert").to(device)
    ref_tokens = tokenizer(reference, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
    cand_tokens = tokenizer(candidate, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
    ref_tokens = {k: v.to(device) for k, v in ref_tokens.items()}
    cand_tokens = {k: v.to(device) for k, v in cand_tokens.items()}

    with torch.no_grad():
        ref_embeddings = model(**ref_tokens).last_hidden_state
        cand_embeddings = model(**cand_tokens).last_hidden_state

    ref_embeddings = ref_embeddings.squeeze(0)
    cand_embeddings = cand_embeddings.squeeze(0)
    cosine_sim = torch.nn.functional.cosine_similarity(ref_embeddings.unsqueeze(1), cand_embeddings.unsqueeze(0), dim=-1)

    ref_to_cand = cosine_sim.max(dim=1).values
    cand_to_ref = cosine_sim.max(dim=0).values

    precision = ref_to_cand.mean().item()
    recall = cand_to_ref.mean().item()

    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return f1_score


def calculate_mathbertscore(reference, candidate, device):
    tokenizer = AutoTokenizer.from_pretrained("mathbert")
    model = AutoModel.from_pretrained("mathbert").to(device)
    ref_tokens = tokenizer(reference, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
    cand_tokens = tokenizer(candidate, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
    ref_tokens = {k: v.to(device) for k, v in ref_tokens.items()}
    cand_tokens = {k: v.to(device) for k, v in cand_tokens.items()}

    with torch.no_grad():
        ref_embeddings = model(**ref_tokens).last_hidden_state
        cand_embeddings = model(**cand_tokens).last_hidden_state

    ref_embeddings = ref_embeddings.squeeze(0)
    cand_embeddings = cand_embeddings.squeeze(0)
    cosine_sim = torch.nn.functional.cosine_similarity(ref_embeddings.unsqueeze(1), cand_embeddings.unsqueeze(0), dim=-1)

    ref_to_cand = cosine_sim.max(dim=1).values
    cand_to_ref = cosine_sim.max(dim=0).values

    precision = ref_to_cand.mean().item()
    recall = cand_to_ref.mean().item()

    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return f1_score
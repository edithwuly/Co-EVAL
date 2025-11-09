import torch
from sklearn.metrics import euclidean_distances
from pyemd import emd
from transformers import AutoTokenizer, AutoModel


def calculate_movers_score(reference, candidate, device):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    # Tokenize the input texts
    ref_tokens = tokenizer(reference, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(
        device)
    cand_tokens = tokenizer(candidate, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(
        device)

    # Get the embeddings for the reference and candidate
    with torch.no_grad():
        ref_embeddings = model(**ref_tokens).last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
        cand_embeddings = model(**cand_tokens).last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

    # Squeeze out the batch dimension and get the token-level embeddings (remove padding tokens)
    ref_embeddings = ref_embeddings.squeeze(0)  # Shape: (seq_len, hidden_dim)
    cand_embeddings = cand_embeddings.squeeze(0)  # Shape: (seq_len, hidden_dim)

    # Ensure embeddings are not padded and match lengths
    ref_tokens_len = ref_tokens['attention_mask'].sum().item()
    cand_tokens_len = cand_tokens['attention_mask'].sum().item()

    ref_embeddings = ref_embeddings[:ref_tokens_len]
    cand_embeddings = cand_embeddings[:cand_tokens_len]

    # Calculate the pairwise Euclidean distance matrix between the embeddings
    distance_matrix = torch.cdist(ref_embeddings, cand_embeddings, p=2).cpu().detach().numpy().astype('float64')

    # Define weights (uniform distribution over the sequence length)
    len1 = ref_embeddings.size(0)
    len2 = cand_embeddings.size(0)

    weights1 = (torch.ones(len1) / len1).numpy().astype('float64')
    weights2 = (torch.ones(len2) / len2).numpy().astype('float64')

    # Compute the Earth Mover's Distance (EMD)
    score = emd(weights1, weights2, distance_matrix)

    return score

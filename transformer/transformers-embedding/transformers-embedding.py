import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    # Your code here
    embeddings=nn.Embedding(vocab_size,d_model)
    return embeddings
    pass

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # Your code here
    embedded=embedding(tokens)
    scaled_embed=embedded*math.sqrt(d_model)
    return scaled_embed
    pass
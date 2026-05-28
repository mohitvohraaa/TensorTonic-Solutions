import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V):

    dk = math.sqrt(K.shape[-1])

    scores = torch.matmul(Q,K.transpose(-2, -1))

    attention_weights = F.softmax(scores / dk, dim=-1)

    output = torch.matmul(attention_weights, V)

    return output
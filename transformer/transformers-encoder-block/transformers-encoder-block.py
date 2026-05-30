import numpy as np
import math
def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    u=np.mean(x,axis=-1,keepdims=True)
    var=np.var(x,axis=-1,keepdims=True)
    x=x-u
    x=x/np.sqrt(var+eps)
    return gamma * x + beta
    
    pass

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    Q_dash=Q @ W_q
    K_dash=K @ W_k
    V_dash=V @ W_v
    final_heads=[]
    dk=K.shape[-1]//num_heads
    for i in range(0,K.shape[-1],dk):
        Qi=Q_dash[...,i:i+dk]
        Ki=K_dash[...,i:i+dk]
        Vi=V_dash[...,i:i+dk]
        S=Qi @ np.swapaxes(Ki,-1,-2)
        S=S/np.sqrt(dk)
        Si=softmax(S)
        head= Si @ Vi
        final_heads.append(head)
        
    return np.concatenate(final_heads,axis=-1) @ W_o
    pass

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    return np.maximum(0,(x @ W1)+b1)@W2 + b2
    
    pass

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    attn=multi_head_attention(
        x,x,x,
        W_q,W_k,W_v,
        W_o,num_heads
        
    )
    x=layer_norm(
        x+attn,gamma1,beta1
    )
    ffn=feed_forward(x,W1,b1,W2,b2)
    return layer_norm(x+ffn,gamma2,beta2)
    
    pass
import numpy as np
import math

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    Q1 = Q @ W_q
    K1 = K @ W_k
    V1 = V @ W_v
    dk=K1.shape[-1]//num_heads
    heads=[]
    for i in range(0,K1.shape[-1],dk):
        Q_dash=Q1[...,i:i+dk]
        K_dash=K1[...,i:i+dk]
        V_dash=V1[...,i:i+dk]
        S=Q_dash @ np.swapaxes(K_dash, -1, -2)
        S1=S/math.sqrt(dk)
        S1_dash=softmax(S1)
        head=S1_dash @ V_dash
        heads.append(head)
        
        
    final_heads=np.concatenate(heads,axis=-1)   
    return final_heads @ W_o
    
    
    pass
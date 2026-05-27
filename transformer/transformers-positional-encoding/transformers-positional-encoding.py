import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    
    
    
    ans=np.zeros((seq_length,d_model))
    for i in range(seq_length):
        
        for j in range(0,d_model,2):
            
            curr_div=10000**(j/d_model)
            ans[i,j]=np.sin(i/curr_div)
            
            if(j+1<d_model):
                ans[i,j+1]=np.cos(i/curr_div)
            
            
            
            
          
    return ans
    pass
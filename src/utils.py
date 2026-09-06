import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """
    Locks all random number generators to ensure complete reproducibility 
    across CPU and CUDA environments.
    """
    # Standard Python and environment seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy seed
    np.random.seed(seed)
    
    # PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Force deterministic algorithms in cuDNN 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import torch
import numpy as np
import random
import os

# Define the standard Sentinel-2 bands order used across all experiments
BANDS_ORDER = [
    "B01", "B02", "B03", "B04",
    "B05", "B06", "B07", "B08",
    "B8A", "B09", "B11", "B12"
]

def set_seed(seed=42):
    """
    Locks all random number generators to ensure complete reproducibility 
    across CPU and CUDA environments.
    """
    # 1. Set standard Python random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Set NumPy random seed
    np.random.seed(seed)
    
    # 3. Set PyTorch random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # 4. Force deterministic algorithms in cuDNN 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

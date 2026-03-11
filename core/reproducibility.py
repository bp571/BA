"""
Reproduzierbarkeitsfunktionen für deterministisches Modellverhalten.
"""
import torch
import numpy as np
import random


def set_all_seeds(seed=13):
    """
    Setzt alle relevanten Random Seeds für maximale Reproduzierbarkeit.
    
    Dies gewährleistet deterministisches Verhalten für:
    - Pythons eingebautes Random-Modul
    - NumPy-Operationen
    - PyTorch CPU-Operationen
    - PyTorch CUDA-Operationen (falls verfügbar)
    
    Args:
        seed (int): Random Seed Wert. Standard: 13
    
    Hinweis:
        Die Verwendung deterministischer CUDNN-Algorithmen kann die Performance
        leicht reduzieren, gewährleistet aber Reproduzierbarkeit über verschiedene Läufe hinweg.
    """
    # Python Random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # PyTorch CUDA (falls verfügbar)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Für Multi-GPU-Setups
    
    # Gewährleiste deterministisches Verhalten in CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ Random Seeds auf {seed} gesetzt für Reproduzierbarkeit")

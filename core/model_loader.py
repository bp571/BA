"""
Schlanke Model Loader für Chronos und Kronos - BA Projekt
"""

import torch
import sys
from pathlib import Path
from typing import Optional, Tuple

# Project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'models' / 'Kronos'))


class ChronosLoader:
    """Schlanker Loader für Chronos Modelle"""
    
    @staticmethod
    def load(model_name: str = "amazon/chronos-2", device_map: str = "auto"):
        from chronos import Chronos2Pipeline
        return Chronos2Pipeline.from_pretrained(model_name, device_map=device_map)


class KronosLoader:
    """Schlanker Loader für Kronos Modelle"""
    
    @staticmethod
    def load(cache_dir: Optional[str] = None, device: Optional[str] = None):
        from model.kronos import KronosPredictor, Kronos, KronosTokenizer
        
        if cache_dir is None:
            cache_dir = str(project_root / 'models' / 'model_cache')
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        
        # Load model and tokenizer
        tokenizer = KronosTokenizer.from_pretrained(
            "NeoQuasar/Kronos-Tokenizer-base", cache_dir=cache_dir
        )
        model = Kronos.from_pretrained(
            "NeoQuasar/Kronos-base", cache_dir=cache_dir
        )
        
        # Setup for inference
        model = model.to(device)
        tokenizer = tokenizer.to(device)
        model.eval()
        tokenizer.eval()
        
        return model, tokenizer
    
    @staticmethod
    def get_predictor(cache_dir: Optional[str] = None, device: Optional[str] = None):
        from model.kronos import KronosPredictor
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        
        model, tokenizer = KronosLoader.load(cache_dir, device)
        
        return KronosPredictor(model=model, tokenizer=tokenizer, device=device)


# Backward compatibility - basierend auf der existierenden load_kronos_model Funktion
def load_kronos_model(cache_dir: Optional[str] = None):
    """Kompatibilität mit existierendem Code"""
    return KronosLoader.load(cache_dir)
import torch
import sys
from pathlib import Path

# Projekt-Wurzelverzeichnis (eine Ebene hoch von 'core')
project_root = Path(__file__).resolve().parent.parent

# Robust path handling for Kronos model imports
kronos_repo_path = project_root / 'models' / 'Kronos'
if not kronos_repo_path.exists():
    raise FileNotFoundError(f"Kronos model directory not found at: {kronos_repo_path}")

if str(kronos_repo_path) not in sys.path:
    sys.path.insert(0, str(kronos_repo_path))

def load_kronos_predictor(device=None, cache_dir=None):
    """Lädt Kronos direkt als einsatzbereiten Predictor"""
    # Diese Imports müssen innerhalb der Funktion stehen, 
    # damit sys.path vorher angepasst werden kann
    from model.kronos import Kronos, KronosTokenizer, KronosPredictor
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cache_dir is None:
        cache_dir = str(project_root / 'models' / 'model_cache')

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", cache_dir=cache_dir)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base", cache_dir=cache_dir).to(device).eval()
    
    return KronosPredictor(model=model, tokenizer=tokenizer, device=device, max_context=512)

# Für Chronos (falls benötigt)
def load_chronos(model_name="amazon/chronos-2", device="cuda"):
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(model_name, device_map=device)

def load_chronos_predictor(model_name="amazon/chronos-2", device=None, cache_dir=None):
    """
    Lädt Chronos2 als einsatzbereiten Predictor mit standardisierter API.
    
    Args:
        model_name: Chronos2 model name (default: "amazon/chronos-2")
        device: torch device string ("cuda" oder "cpu", None für auto-detect)
        cache_dir: Cache-Verzeichnis für Modell-Downloads
    
    Returns:
        ChronosPredictor Instanz mit standardisierter Predictor-API
    """
    from core.chronos_wrapper import ChronosPredictor
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Lade Chronos2 Pipeline
    pipeline = load_chronos(model_name=model_name, device=device)
    
    return ChronosPredictor(pipeline=pipeline, device=device)
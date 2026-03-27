import torch
import sys
from pathlib import Path
from peft import PeftModel  

# Projekt-Wurzelverzeichnis
project_root = Path(__file__).resolve().parent.parent

# Pfadbehandlung für Kronos
kronos_repo_path = project_root / '02_finetuning' / 'models' / 'Kronos'
if str(kronos_repo_path) not in sys.path:
    sys.path.insert(0, str(kronos_repo_path))

def load_chronos(model_name="amazon/chronos-2", device="cuda", adapter_path=None):
    """
    Erweiterte Ladefunktion für Chronos2 mit optionalem LoRA-Adapter Support.
    """
    from chronos import Chronos2Pipeline
    
    # 1. Lade die Standard-Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(model_name, device_map=device)
    
    # 2. Falls ein Adapter-Pfad angegeben ist, injiziere die LoRA-Gewichte
    if adapter_path:
        print(f"Lade LoRA-Adapter von: {adapter_path}")
        # Die Pipeline hält das Modell in self.model
        pipeline.model = PeftModel.from_pretrained(pipeline.model, adapter_path)
        # Optional: Mergen für schnellere Inferenz
        # pipeline.model = pipeline.model.merge_and_unload()
        
    return pipeline

def load_chronos_predictor(model_name="amazon/chronos-2", device=None, cache_dir=None, adapter_path=None):
    """
    Erweitert den Predictor um die Fähigkeit, Fine-Tuning-Gewichte zu nutzen.
    """
    from core.chronos_wrapper import ChronosPredictor
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Lade Pipeline (mit oder ohne Adapter)
    pipeline = load_chronos(model_name=model_name, device=device, adapter_path=adapter_path)
    
    return ChronosPredictor(pipeline=pipeline, device=device)

def load_kronos_predictor(device=None, cache_dir=None, adapter_path=None):
    """Lädt Kronos direkt als einsatzbereiten Predictor."""
    from model.kronos import Kronos, KronosTokenizer, KronosPredictor
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cache_dir is None:
        cache_dir = str(project_root / '02_finetuning' / 'models' / 'model_cache')

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", cache_dir=cache_dir)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base", cache_dir=cache_dir)
    
    if adapter_path:
        from peft import PeftModel
        print(f"Loading Kronos LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.to(device).eval()
    return KronosPredictor(model=model, tokenizer=tokenizer, device=device, max_context=512)


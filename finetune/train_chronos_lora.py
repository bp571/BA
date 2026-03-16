"""
Chronos-2 LoRA Fine-Tuning Script
Direktes Training mit dem Chronos2Trainer API
"""
import sys
from pathlib import Path

# Projekt-Root zum Path hinzufügen
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from transformers import TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model
from chronos.chronos2.model import Chronos2Model
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from chronos.chronos2.trainer import Chronos2Trainer, EvaluateAndSaveFinalStepCallback
from gluonts.dataset.arrow import ArrowFile


def load_arrow_dataset(path: Path):
    """Lädt ein Arrow-Dataset und konvertiert es für Chronos2Dataset."""
    print(f"📂 Lade Daten von: {path}")
    
    # Lade Arrow-Datei
    arrow_file = ArrowFile(path)
    dataset = list(arrow_file)
    
    print(f"   Gefunden: {len(dataset)} Zeitreihen")
    
    # Konvertiere zu Chronos2-Format
    inputs = []
    for entry in dataset:
        # entry hat: 'start', 'target', 'item_id'
        target = torch.tensor(entry["target"], dtype=torch.float32)
        inputs.append({"target": target})
    
    return inputs


class Chronos2LoRATrainer(Chronos2Trainer):
    """
    Custom Trainer für Chronos2 mit LoRA-Support.
    Überschreibt compute_loss um explizit mit den richtigen Inputs zu arbeiten.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Berechnet Loss mit korrekten Chronos2-Inputs.
        """
        # Entferne die Keys, die nicht vom Modell erwartet werden
        inputs_copy = {k: v for k, v in inputs.items() if k != "target_idx_ranges"}
        
        # Rufe das Modell mit den richtigen Parametern auf
        outputs = model(**inputs_copy)
        
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def train():
    """Hauptfunktion für LoRA Fine-Tuning."""
    
    # =====================
    # 1. KONFIGURATION
    # =====================
    config = {
        "model_id": "amazon/chronos-2",
        "context_length": 512,
        "prediction_length": 12,
        "batch_size": 8,
        "max_steps": 1000,
        "learning_rate": 1e-4,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 2,
        "save_steps": 250,
        "logging_steps": 10,
        "eval_steps": 125,
        "seed": 42,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q", "v", "k", "o"],
    }
    
    # Pfade
    data_path = project_root / "data" / "processed" / "train_data.arrow"
    val_data_path = project_root / "data" / "processed" / "val_data.arrow"
    output_dir = project_root / "models" / "chronos-2-lora-finetuned"
    log_dir = project_root / "logs" / "chronos-training"
    
    # Erstelle Ausgabeverzeichnisse
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Validierung
    if not data_path.exists():
        raise FileNotFoundError(
            f"❌ Trainingsdaten nicht gefunden: {data_path}\n"
            f"Bitte führe zuerst 'python finetune/prepare_data.py' aus."
        )
    
    # GPU-Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    if device == "cpu":
        print("⚠️  Warnung: Kein CUDA verfügbar, Training läuft auf CPU (langsam!)")
    
    # Seed setzen
    set_seed(config["seed"])
    
    # =====================
    # 2. MODELL LADEN
    # =====================
    print(f"\n📥 Lade Modell: {config['model_id']}")
    model = Chronos2Model.from_pretrained(config["model_id"])
    
    # LoRA-Konfiguration
    print(f"\n🔧 Konfiguriere LoRA:")
    print(f"   Rank: {config['lora_r']}")
    print(f"   Alpha: {config['lora_alpha']}")
    print(f"   Target Modules: {config['lora_target_modules']}")
    
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=None,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Workaround: Füge fehlende Methode hinzu für HuggingFace Trainer-Kompatibilität
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Dummy-Methode für Trainer-Kompatibilität."""
        return {}
    
    # Binde die Methode an das Modell
    import types
    model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, model)
    
    # =====================
    # 3. DATEN LADEN
    # =====================
    train_inputs = load_arrow_dataset(data_path)
    
    # Zugriff auf chronos_config über das base model bei PEFT
    base_model = model.base_model if hasattr(model, 'base_model') else model
    output_patch_size = base_model.chronos_config.output_patch_size
    
    train_dataset = Chronos2Dataset(
        inputs=train_inputs,
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        batch_size=config["batch_size"],
        output_patch_size=output_patch_size,
        mode=DatasetMode.TRAIN,
    )
    
    eval_dataset = None
    if val_data_path.exists():
        print(f"📂 Lade Validierungsdaten von: {val_data_path}")
        val_inputs = load_arrow_dataset(val_data_path)
        eval_dataset = Chronos2Dataset(
            inputs=val_inputs,
            context_length=config["context_length"],
            prediction_length=config["prediction_length"],
            batch_size=config["batch_size"],
            output_patch_size=output_patch_size,
            mode=DatasetMode.VALIDATION,
        )
    else:
        print("⚠️  Keine Validierungsdaten gefunden")
    
    # =====================
    # 4. TRAINING SETUP
    # =====================
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(log_dir),
        max_steps=config["max_steps"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=1,  # Dataset gibt bereits Batches zurück
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"] if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=False if eval_dataset else False,
        metric_for_best_model=None if eval_dataset else None,
        greater_is_better=False,
        report_to="tensorboard",
        seed=config["seed"],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,  # Dataset ist bereits im Speicher
        remove_unused_columns=False,
        prediction_loss_only=True,  # Verhindert Aufruf von prepare_inputs_for_generation
    )
    
    # =====================
    # 5. TRAINER ERSTELLEN
    # =====================
    trainer = Chronos2LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EvaluateAndSaveFinalStepCallback()],
    )
    
    # =====================
    # 6. TRAINING STARTEN
    # =====================
    print(f"\n🚀 Starte Training:")
    print(f"   Max Steps: {config['max_steps']}")
    print(f"   Effektive Batch Size: {config['batch_size']} x {config['gradient_accumulation_steps']} = {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Output: {output_dir}")
    print(f"\n💡 Monitoring mit: tensorboard --logdir {log_dir}\n")
    
    trainer.train()
    
    # =====================
    # 7. MODELL SPEICHERN
    # =====================
    print(f"\n💾 Speichere finales Modell...")
    trainer.save_model(str(output_dir / "final"))
    
    print(f"\n✅ Training abgeschlossen!")
    print(f"   Modell gespeichert in: {output_dir / 'final'}")

if __name__ == "__main__":
    train()

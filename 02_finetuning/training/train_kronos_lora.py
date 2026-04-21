import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed
from peft import LoraConfig, get_peft_model
from gluonts.dataset.arrow import ArrowFile
import numpy as np

kronos_repo_path = project_root / 'models' / 'Kronos'
if str(kronos_repo_path) not in sys.path:
    sys.path.insert(0, str(kronos_repo_path))

from model.kronos import Kronos, KronosTokenizer


class KronosDataset(Dataset):
    def __init__(self, arrow_path, context_length=512):
        self.context_length = context_length
        arrow_file = ArrowFile(arrow_path)
        self.entries = list(arrow_file)
        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        target = entry['target']  # Shape: (seq_len, 6) for OHLCV+Amount
        
        if len(target) > self.context_length:
            start_idx = np.random.randint(0, len(target) - self.context_length)
            target = target[start_idx:start_idx + self.context_length]
        
        # target already has shape (seq_len, 6): [open, high, low, close, volume, amount]
        x = target
        
        # Daily data: minute=0, hour=0, weekday/day/month from index
        timestamps = np.arange(len(target))
        time_features = np.stack([
            np.zeros(len(target)),       # minute: 0
            np.zeros(len(target)),       # hour: 0
            timestamps % 7,              # weekday: 0-6
            timestamps % 31 + 1,         # day: 1-31
            (timestamps // 30) % 12 + 1  # month: 1-12
        ], axis=-1)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(time_features, dtype=torch.float32)


def train():
    config = {
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "context_length": 512,
        "batch_size": 4,
        "max_steps": 1000,
        "learning_rate": 1e-4,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 2,
        "save_steps": 250,
        "logging_steps": 10,
        "seed": 42,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    }
    
    data_path = project_root / "data" / "processed" / "train_data_kronos.arrow"
    output_dir = project_root / "models" / "kronos-lora-finetuned"
    cache_dir = project_root / "models" / "model_cache"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    set_seed(config["seed"])
    
    print(f"\nLoading tokenizer: {config['tokenizer_id']}")
    tokenizer = KronosTokenizer.from_pretrained(config["tokenizer_id"], cache_dir=str(cache_dir))
    tokenizer.eval().to(device)
    
    print(f"Loading model: {config['model_id']}")
    model = Kronos.from_pretrained(config["model_id"], cache_dir=str(cache_dir))
    
    target_modules = []
    n_layers = len(model.transformer)
    for i in range(n_layers):
        target_modules.extend([
            f"transformer.{i}.self_attn.q_proj",
            f"transformer.{i}.self_attn.k_proj",
            f"transformer.{i}.self_attn.v_proj",
            f"transformer.{i}.self_attn.out_proj"
        ])
    
    print(f"\nApplying LoRA:")
    print(f"  Rank: {config['lora_r']}")
    print(f"  Alpha: {config['lora_alpha']}")
    print(f"  Target modules: {len(target_modules)}")
    
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=None
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    
    dataset = KronosDataset(data_path, config["context_length"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    total_steps = config["max_steps"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        total_steps=total_steps,
        pct_start=min(0.3, config["warmup_steps"] / total_steps)
    )
    
    print(f"\nTraining:")
    print(f"  Max steps: {config['max_steps']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"  Output: {output_dir}\n")
    
    model.train()
    global_step = 0
    optimizer.zero_grad()
    
    while global_step < config["max_steps"]:
        for batch_x, batch_x_stamp in dataloader:
            batch_x = batch_x.to(device)
            batch_x_stamp = batch_x_stamp.to(device)
            
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            
            s1_logits, s2_logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            
            loss, s1_loss, s2_loss = model.base_model.head.compute_loss(
                s1_logits, s2_logits,
                token_out[0], token_out[1]
            )
            
            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()
            
            if (global_step + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if (global_step + 1) % config["logging_steps"] == 0:
                print(f"Step {global_step + 1}/{config['max_steps']}: Loss={loss.item():.4f}, S1={s1_loss.item():.4f}, S2={s2_loss.item():.4f}")
            
            if (global_step + 1) % config["save_steps"] == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step + 1}"
                model.save_pretrained(str(checkpoint_dir))
                print(f"Checkpoint saved: {checkpoint_dir}")
            
            global_step += 1
            if global_step >= config["max_steps"]:
                break
    
    print(f"\nSaving final model...")
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    print(f"Training complete! Model saved to: {final_dir}")


if __name__ == "__main__":
    train()

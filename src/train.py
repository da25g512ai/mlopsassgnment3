import os
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from dataset import get_dataloaders
from model import get_model

def load_config(config_path_str: str | None = None) -> dict:
    paths = [config_path_str, os.getenv("CONFIG_PATH"), "/app/configs/training_config.yaml", "configs/training_config.yaml"]
    for p in paths:
        if p and Path(p).exists():
            with open(p, "r") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("Config file not found.")

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, scheduler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config["model"]["architecture"], config["model"]["num_classes"]).to(device)
    # if int(torch.__version__.split('.')[0]) >= 2 and device.type == "cuda":
    #     model = torch.compile(model)
    train_loader, val_loader = get_dataloaders(config["data"]["data_dir"], config["training"]["batch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["training"]["learning_rate"], 
        epochs=config["training"]["epochs"], steps_per_epoch=len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda') 
    best_val_loss, patience_counter = float("inf"), 0
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, scheduler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(json.dumps({
            "epoch": epoch + 1, "train_loss": round(train_loss, 4), "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4), "val_accuracy": round(val_acc, 4)
        }), flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / config["output"]["model_name"]
            torch.save({"model_state_dict": model.state_dict()}, save_path)
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                break

if __name__ == "__main__":
    main()
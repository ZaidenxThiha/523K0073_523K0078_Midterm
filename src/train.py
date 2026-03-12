from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .evaluate import accuracy_from_logits, evaluate


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, total_acc, total_samples = 0.0, 0.0, 0

    for batch_x, batch_y in tqdm(dataloader, desc="train", leave=False):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        total_acc += accuracy_from_logits(logits, batch_y) * len(batch_y)
        total_samples += len(batch_y)

    return total_loss / total_samples, total_acc / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, list]:
    """
    Train model and optionally save the best (by val_acc) checkpoint to checkpoint_path.
    Returns a history dict of losses/accuracies.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if ckpt_path:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)

        print(
            f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.4f} | best val acc {best_val_acc:.4f}"
        )

    history["best_val_acc"] = best_val_acc
    return history

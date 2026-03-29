from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .evaluate import evaluate_loader, plot_learning_curves, top_misclassified_examples
    from .preprocess import prepare_recurrent_data
    from .rnn_model import DynamicPadCollator, RNNClassifier, TextDataset
except ImportError:  # pragma: no cover - fallback for direct script execution
    from evaluate import evaluate_loader, plot_learning_curves, top_misclassified_examples
    from preprocess import prepare_recurrent_data
    from rnn_model import DynamicPadCollator, RNNClassifier, TextDataset


RANDOM_SEED = 42
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
FIGURES_DIR = CHECKPOINT_DIR / "figures"


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed PyTorch, NumPy, and Python RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    """Create output directories needed for saving checkpoints."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def build_loaders(
    vocab_size: int = 30000,
    vocab_sizes: list[int] | None = None,
    batch_size: int = 64,
    max_len: int = 455,
):
    """Build train, validation, and test loaders for the selected recurrent setup."""
    preprocessing = prepare_recurrent_data(
        max_len=max_len,
        vocab_sizes=vocab_sizes,
        selected_vocab_size=vocab_size,
        seed=RANDOM_SEED,
    )
    encoded_splits = preprocessing["selected_encoded_splits"]
    vocab = preprocessing["selected_vocab"]
    collator = DynamicPadCollator()
    train_dataset = TextDataset(encoded_splits["train"]["encoded_ids"].tolist(), encoded_splits["train"]["label"].tolist(), encoded_splits["train"]["clean_text"].tolist())
    val_dataset = TextDataset(encoded_splits["validation"]["encoded_ids"].tolist(), encoded_splits["validation"]["label"].tolist(), encoded_splits["validation"]["clean_text"].tolist())
    test_dataset = TextDataset(encoded_splits["test"]["encoded_ids"].tolist(), encoded_splits["test"]["label"].tolist(), encoded_splits["test"]["clean_text"].tolist())
    return preprocessing, vocab, (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator),
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one full training or evaluation pass over a dataloader."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    progress = tqdm(loader, leave=False, disable=not is_train)
    for input_ids, lengths, labels, _texts in progress:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        if is_train:
            optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_items += batch_size
    return total_loss / total_items, total_correct / total_items


def train_model(
    variant: str,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 1,
    epochs: int = 3,
    dropout: float = 0.5,
    vocab_size: int = 30000,
    vocab_sizes: list[int] | None = None,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_len: int = 455,
    device: str | torch.device = "cpu",
    checkpoint_name: str | None = None,
    curve_title: str | None = None,
    figure_name: str | None = None,
):
    """Train one recurrent model variant and return metrics, artifacts, and predictions."""
    ensure_dirs()
    device = torch.device(device)
    preprocessing, vocab, (train_loader, val_loader, test_loader) = build_loaders(
        vocab_size=vocab_size,
        vocab_sizes=vocab_sizes,
        batch_size=batch_size,
        max_len=max_len,
    )
    model = RNNClassifier(len(vocab), embed_dim, hidden_dim, num_layers=num_layers, rnn_type=variant, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_val_acc = 0.0
    checkpoint_path = CHECKPOINT_DIR / (checkpoint_name or f"{variant}_best.pt")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = evaluate_loader(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate_loader(model, test_loader, criterion, device)
    training_time = time.time() - start_time
    history_df = pd.DataFrame(history)
    learning_curve_figure = plot_learning_curves(history_df, title=curve_title or variant.upper())
    figure_path = FIGURES_DIR / (figure_name or f"{variant}_learning_curves.png")
    learning_curve_figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    misclassified_df = top_misclassified_examples(test_metrics["rows"])

    return {
        "model": variant.upper() if variant != "rnn" else "Vanilla RNN",
        "variant": variant,
        "checkpoint_path": str(checkpoint_path),
        "epochs": epochs,
        "vocab_size": vocab_size,
        "max_len": max_len,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "training_time_sec": training_time,
        "time_per_epoch_sec": training_time / max(epochs, 1),
        "history_df": history_df,
        "curve_figure": learning_curve_figure,
        "figure_path": str(figure_path),
        "misclassified_df": misclassified_df,
        "predictions": pd.DataFrame(test_metrics["rows"]),
        "split_summary_df": preprocessing["split_summary_df"],
        "vocab_summary_df": preprocessing["vocab_summary_df"],
    }


def train_all_three_models(
    epochs: int = 3,
    vocab_size: int = 30000,
    vocab_sizes: list[int] | None = None,
    max_len: int = 455,
    device: str | torch.device = "cpu",
):
    """Train the vanilla RNN, LSTM, and GRU variants under one shared configuration."""
    return [
        train_model(
            variant,
            epochs=epochs,
            vocab_size=vocab_size,
            vocab_sizes=vocab_sizes,
            max_len=max_len,
            device=device,
            checkpoint_name=f"{variant}_best.pt",
            figure_name=f"{variant}_learning_curves.png",
        )
        for variant in ["rnn", "lstm", "gru"]
    ]


def result_to_export_row(result: dict[str, object]) -> dict[str, object]:
    """Convert a training result payload to a notebook-friendly export row."""
    return {
        "model": result["model"],
        "variant": result["variant"],
        "accuracy": result["accuracy"],
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": result["f1"],
        "time_per_epoch_sec": result["time_per_epoch_sec"],
        "checkpoint_path": result["checkpoint_path"],
        "figure_path": result["figure_path"],
    }


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a PyTorch module."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

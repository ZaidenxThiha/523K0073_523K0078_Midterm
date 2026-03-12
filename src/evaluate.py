from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float().sum().item()
    return correct / len(labels)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss, total_acc, total_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="eval", leave=False):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_y)
            total_acc += accuracy_from_logits(logits, batch_y) * len(batch_y)
            total_samples += len(batch_y)
    return total_loss / total_samples, total_acc / total_samples


def predict(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the model over a dataloader and collect logits/preds/labels on CPU for analysis.
    """
    model.eval()
    logits_list, preds_list, labels_list = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="predict", leave=False):
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=1)
            logits_list.append(logits.cpu())
            preds_list.append(preds.cpu())
            labels_list.append(batch_y.cpu())

    return torch.cat(logits_list), torch.cat(preds_list), torch.cat(labels_list)


def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """
    Compute accuracy, classification report, and confusion matrix from logits/labels.
    target_names (e.g., ["neg", "pos"]) makes reports easier to read.
    """
    y_true = labels.cpu().numpy()
    y_pred = logits.argmax(dim=1).cpu().numpy()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def plot_learning_curves(
    history: Dict[str, Iterable[float]],
    save_path: Optional[str] = None,
    title: str = "Learning curves",
):
    """
    Plot train/val loss and accuracy from the history returned by train_model.
    Optionally save the figure to save_path.
    """
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history.get("train_loss", []), label="train")
    axes[0].plot(epochs, history.get("val_loss", []), label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history.get("train_acc", []), label="train")
    axes[1].plot(epochs, history.get("val_acc", []), label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes


def misclassified_examples(
    logits: torch.Tensor,
    labels: torch.Tensor,
    texts: Optional[Sequence[str]] = None,
    limit: Optional[int] = 20,
) -> List[Dict[str, object]]:
    """
    Return a list of misclassified examples for manual error analysis.
    If `texts` is provided, it should align with the dataloader order used to produce logits.
    """
    preds = logits.argmax(dim=1)
    mismatches = (preds != labels).nonzero(as_tuple=False).flatten().tolist()
    results: List[Dict[str, object]] = []

    for idx in mismatches[: limit or len(mismatches)]:
        item: Dict[str, object] = {
            "index": int(idx),
            "pred": int(preds[idx].item()),
            "label": int(labels[idx].item()),
        }
        if texts is not None and idx < len(texts):
            item["text"] = texts[idx]
        results.append(item)

    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    target_names: Sequence[str],
    normalize: bool = True,
    cmap: str = "Blues",
):
    """Plot a confusion matrix with optional normalization."""
    cm_to_plot = cm.astype("float")
    if normalize and cm_to_plot.sum(axis=1, keepdims=True).any():
        cm_to_plot = cm_to_plot / cm_to_plot.sum(axis=1, keepdims=True)

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(
        cm_to_plot,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

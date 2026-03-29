from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate_loader(model, loader, criterion, device):
    """Evaluate a sequence model loader and return aggregate metrics plus per-row outputs."""
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_labels = []
    all_preds = []
    all_rows = []

    with torch.no_grad():
        for input_ids, lengths, labels, texts in loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
            labels_cpu = labels.cpu().tolist()
            preds_cpu = preds.cpu().tolist()
            probs_cpu = probs.cpu().tolist()
            all_labels.extend(labels_cpu)
            all_preds.extend(preds_cpu)
            for text, true_label, pred_label, prob_pair in zip(texts, labels_cpu, preds_cpu, probs_cpu):
                all_rows.append(
                    {
                        "text": text,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "confidence_negative": float(prob_pair[0]),
                        "confidence_positive": float(prob_pair[1]),
                        "is_correct": int(true_label == pred_label),
                    }
                )

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    return {
        "loss": total_loss / total_items,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rows": all_rows,
    }


def plot_learning_curves(history_df: pd.DataFrame, title: str):
    """Plot train/validation loss and accuracy from an epoch-by-epoch history frame."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Validation loss")
    axes[0].set_title(f"{title}: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], marker="o", label="Train accuracy")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], marker="o", label="Validation accuracy")
    axes[1].set_title(f"{title}: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    return fig


def top_misclassified_examples(rows: list[dict], max_examples: int = 10) -> pd.DataFrame:
    """Return the highest-confidence misclassified predictions."""
    errors = [row for row in rows if not row["is_correct"]]
    errors = sorted(errors, key=lambda row: max(row["confidence_negative"], row["confidence_positive"]), reverse=True)
    return pd.DataFrame(errors[:max_examples])


def build_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Format a model-comparison DataFrame into the report summary table layout."""
    summary_df = results_df.copy()
    summary_df["Model"] = summary_df["model"]
    summary_df["Acc."] = summary_df["accuracy"].round(4)
    summary_df["Prec."] = summary_df["precision"].round(4)
    summary_df["Recall"] = summary_df["recall"].round(4)
    summary_df["F1"] = summary_df["f1"].round(4)
    summary_df["Time/epoch"] = summary_df["time_per_epoch_sec"].round(2)
    return summary_df[["Model", "Acc.", "Prec.", "Recall", "F1", "Time/epoch"]]

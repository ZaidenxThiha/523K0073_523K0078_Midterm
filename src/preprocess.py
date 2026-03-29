from __future__ import annotations

import random
import re
from collections import Counter

import numpy as np
import pandas as pd
from datasets import DatasetDict, concatenate_datasets, load_dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
TOKEN_PATTERN = re.compile(r"[A-Za-z']+")
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed the Python and NumPy RNGs for reproducible preprocessing."""
    random.seed(seed)
    np.random.seed(seed)


def clean_text(text: str) -> str:
    """Lowercase review text and remove HTML fragments and non-letter symbols."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text into alphabetic word tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def summarize_lengths(lengths: list[int]) -> dict[str, float]:
    """Return descriptive statistics for a list of tokenized sequence lengths."""
    arr = np.array(lengths)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def create_stratified_splits(seed: int = RANDOM_SEED) -> DatasetDict:
    """Load IMDb and create deterministic 70/10/20 stratified train/val/test splits."""
    dataset = load_dataset("imdb")
    full_dataset = concatenate_datasets([dataset["train"], dataset["test"]])
    train_split = full_dataset.train_test_split(test_size=0.3, stratify_by_column="label", seed=seed)
    remaining = train_split["test"]
    val_test_split = remaining.train_test_split(test_size=2 / 3, stratify_by_column="label", seed=seed)
    return DatasetDict(
        {
            "train": train_split["train"],
            "validation": val_test_split["train"],
            "test": val_test_split["test"],
        }
    )


def prepare_split_frames(seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    """Convert dataset splits to DataFrames with cleaned text, tokens, and token counts."""
    split_dict = create_stratified_splits(seed=seed)
    split_frames = {name: split.to_pandas() for name, split in split_dict.items()}
    for df in split_frames.values():
        df["clean_text"] = df["text"].map(clean_text)
        df["tokens"] = df["clean_text"].map(tokenize)
        df["token_count"] = df["tokens"].map(len)
    return split_frames


def build_vocab(token_lists: list[list[str]], max_vocab_size: int, min_freq: int = 2) -> dict[str, int]:
    """Build a frequency-sorted vocabulary with reserved pad and unknown tokens."""
    counter = Counter(token for tokens in token_lists for token in tokens)
    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_tokens(tokens: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    """Map tokens to ids and pad or truncate the sequence to ``max_len``."""
    encoded = [vocab.get(token, UNK_ID) for token in tokens[:max_len]]
    if len(encoded) < max_len:
        encoded.extend([PAD_ID] * (max_len - len(encoded)))
    return encoded


def truncate_and_encode_tokens(tokens: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    """Map tokens to ids and truncate to ``max_len`` without padding."""
    return [vocab.get(token, UNK_ID) for token in tokens[:max_len]]


def prepare_recurrent_data(
    max_len: int = 455,
    vocab_sizes: list[int] | None = None,
    selected_vocab_size: int = 30000,
    seed: int = RANDOM_SEED,
    min_freq: int = 2,
) -> dict[str, object]:
    """Prepare recurrent-model inputs for multiple vocabulary sizes and one selected setting."""
    if vocab_sizes is None:
        vocab_sizes = [10000, 20000, 30000]

    split_frames = prepare_split_frames(seed=seed)
    train_tokens = split_frames["train"]["tokens"].tolist()
    length_summary = summarize_lengths(split_frames["train"]["token_count"].tolist())

    all_encoded_splits: dict[int, dict[str, pd.DataFrame]] = {}
    vocabularies: dict[int, dict[str, int]] = {}
    vocab_summary_rows = []

    for vocab_size in vocab_sizes:
        vocab = build_vocab(train_tokens, max_vocab_size=vocab_size, min_freq=min_freq)
        vocabularies[vocab_size] = vocab
        encoded_splits: dict[str, pd.DataFrame] = {}
        for split_name, df in split_frames.items():
            encoded_df = df[["text", "clean_text", "label", "token_count"]].copy()
            encoded_df["encoded_ids"] = df["tokens"].map(lambda tokens: truncate_and_encode_tokens(tokens, vocab, max_len))
            encoded_df["effective_length"] = encoded_df["encoded_ids"].map(len)
            encoded_splits[split_name] = encoded_df
        all_encoded_splits[vocab_size] = encoded_splits
        vocab_summary_rows.append(
            {
                "vocab_size_requested": vocab_size,
                "vocab_size_built": len(vocab),
                "pad_id": PAD_ID,
                "unk_id": UNK_ID,
                "max_len": max_len,
            }
        )

    split_summary_df = pd.DataFrame(
        [
            {
                "split": name,
                "size": int(len(df)),
                "negative": int((df["label"] == 0).sum()),
                "positive": int((df["label"] == 1).sum()),
            }
            for name, df in split_frames.items()
        ]
    )
    vocab_summary_df = pd.DataFrame(vocab_summary_rows)

    return {
        "split_frames": split_frames,
        "length_summary": length_summary,
        "split_summary_df": split_summary_df,
        "vocab_summary_df": vocab_summary_df,
        "vocabularies": vocabularies,
        "encoded_splits_by_vocab": all_encoded_splits,
        "selected_vocab_size": selected_vocab_size,
        "selected_vocab": vocabularies[selected_vocab_size],
        "selected_encoded_splits": all_encoded_splits[selected_vocab_size],
    }

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD = 0
UNK = 1

_token_pattern = re.compile(r"[A-Za-z0-9']+")


def clean_text(text: str) -> str:
    """Lowercase and strip html/special characters."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return _token_pattern.findall(text)


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    def __len__(self) -> int:
        return len(self.stoi)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(tok, UNK) for tok in tokens]


def build_vocab(texts: Iterable[str], max_tokens: int) -> Vocab:
    """Build vocabulary keeping the most frequent tokens."""
    counter: Counter[str] = Counter()
    for txt in texts:
        counter.update(tokenize(clean_text(txt)))

    most_common = counter.most_common(max_tokens - 2)
    stoi = {PAD_TOKEN: PAD, UNK_TOKEN: UNK}
    for idx, (tok, _) in enumerate(most_common, start=2):
        stoi[tok] = idx

    itos = [""] * len(stoi)
    for tok, idx in stoi.items():
        itos[idx] = tok

    return Vocab(stoi=stoi, itos=itos)


def encode_text(
    text: str, vocab: Vocab, max_len: int, pad_value: int = PAD
) -> List[int]:
    tokens = tokenize(clean_text(text))
    token_ids = vocab.encode(tokens)[:max_len]
    if len(token_ids) < max_len:
        token_ids += [pad_value] * (max_len - len(token_ids))
    return token_ids


class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: Vocab, max_len: int):
        self.labels = list(labels)
        self.samples = [encode_text(txt, vocab, max_len) for txt in texts]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.samples[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.stack(labels)


def compute_length_stats(texts: Sequence[str]) -> Dict[str, float]:
    lengths = [len(tokenize(clean_text(t))) for t in texts]
    arr = np.array(lengths)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "count": float(len(arr)),
    }


def top_k_frequencies(texts: Sequence[str], k: int) -> List[Tuple[str, int]]:
    counter: Counter[str] = Counter()
    for txt in texts:
        counter.update(tokenize(clean_text(txt)))
    return counter.most_common(k)


def prepare_imdb_splits(dataset, seed: int = 42, train_frac: float = 0.7, val_frac: float = 0.1):
    """
    Merge IMDb train/test splits then create stratified train/val/test.
    Returns: train_ds, val_ds, test_ds (HF Dataset objects).
    """
    from datasets import concatenate_datasets

    assert "train" in dataset and "test" in dataset
    full = concatenate_datasets([dataset["train"], dataset["test"]])
    first_split = full.train_test_split(test_size=1 - train_frac, stratify_by_column="label", seed=seed)
    remaining = first_split["test"]
    relative_val = val_frac / (1 - train_frac)
    val_test_split = remaining.train_test_split(test_size=1 - relative_val, stratify_by_column="label", seed=seed)
    return first_split["train"], val_test_split["train"], val_test_split["test"]

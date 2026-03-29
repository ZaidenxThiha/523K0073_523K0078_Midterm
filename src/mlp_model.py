from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn

from .preprocess import PAD_ID


class MLPClassifier(nn.Module):
    """Embedding-plus-mean-pooling MLP classifier for sentiment analysis."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dims: Iterable[int],
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)

        layers: List[nn.Module] = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average non-padding token embeddings and classify the pooled sentence vector."""
        emb = self.embedding(x)
        mask = (x != PAD_ID).unsqueeze(2)
        emb = emb * mask
        pooled = emb.sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(pooled)

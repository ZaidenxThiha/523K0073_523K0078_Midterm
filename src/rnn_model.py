from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset


PAD_ID = 0


class TextDataset(Dataset):
    """Dataset of encoded token sequences, labels, and original text strings."""

    def __init__(self, sequences: list[list[int]], labels: list[int], texts: list[str]) -> None:
        self.sequences = sequences
        self.labels = labels
        self.texts = texts

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.sequences[index], self.labels[index], self.texts[index]


class DynamicPadCollator:
    """Pad a batch of variable-length sequences and return their true lengths."""

    def __init__(self, pad_id: int = PAD_ID) -> None:
        self.pad_id = pad_id

    def __call__(self, batch):
        """Pad a batch of variable-length examples and return padded ids, lengths, labels, and texts."""
        sequences, labels, texts = zip(*batch)
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        max_batch_len = int(lengths.max())
        padded = torch.full((len(sequences), max_batch_len), self.pad_id, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded, lengths, torch.tensor(labels, dtype=torch.long), list(texts)


class RNNClassifier(nn.Module):
    """Embedding-based RNN/LSTM/GRU text classifier with packed-sequence handling."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.5,
        pad_idx: int = PAD_ID,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[self.rnn_type]
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=recurrent_dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode padded token ids and classify each sequence from the last hidden state."""
        emb = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output = self.rnn(packed)
        if self.rnn_type == "lstm":
            _, (hidden, _) = output
        else:
            _, hidden = output
        h_last = hidden[-1]
        return self.fc(self.dropout(h_last))

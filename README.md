# Deep Learning Midterm — IMDb Sentiment (Parts A–B)

This repository implements the preprocessing pipeline and MLP baseline for IMDb sentiment classification, ready to run in Jupyter notebooks.

## Structure

- `notebooks/01_eda.ipynb` — dataset loading, cleaning, statistics, and vocabulary exploration.
- `notebooks/02_mlp.ipynb` — preprocessing, dataloaders, mean-pooled MLP, and experiment sweeps (depth, embedding size, dropout).
- `src/preprocess.py` — text cleaning, tokenization, vocabulary building, dataset splitting.
- `src/mlp_model.py` — mean-pooling MLP classifier.
- `src/train.py` — shared training loop utilities.
- `src/evaluate.py` — evaluation metrics, learning curves, and error analysis helpers.
- `requirements.txt` — Python deps (PyTorch, datasets, etc.).

## Quickstart

1. Create and activate a venv  
   `python3 -m venv .venv && source .venv/bin/activate`
2. Install deps  
   `pip install --upgrade pip && pip install -r requirements.txt`
3. Launch Jupyter  
   `jupyter lab` or `jupyter notebook`
4. Open `notebooks/01_eda.ipynb` for EDA and `notebooks/02_mlp.ipynb` for MLP training/experiments. Per-run checkpoints save to `checkpoints/<run>.pt`; the best run (by test acc) is also copied to `checkpoints/mlp_best.pt`.

GPU is optional but recommended. Default random seed is 42 for reproducibility.

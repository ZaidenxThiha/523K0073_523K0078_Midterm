# ID1_ID2_Midterm

Deep Learning midterm project for IMDb sentiment analysis, comparing an MLP baseline with recurrent models implemented in native PyTorch.

## Team Contributions

Team submission:

| Member | Student ID | Contribution |
| --- | --- | --- |
| Thiha Aung | 523K0073 | Preprocessing, MLP experiments, notebook integration, README |
| Thin Lei Sandi | 523K0078 | Recurrent-model experiments, comparative analysis, report preparation |

## Project Structure

- `notebooks/01_eda.ipynb`: Part A EDA and preprocessing inspection
- `notebooks/02_mlp.ipynb`: Part B MLP experiments and best-model export
- `notebooks/03_rnn.ipynb`: Part C recurrent-model experiments, ablations, and exports
- `notebooks/04_analysis.ipynb`: Part D comparison, figures, and error-analysis workspace
- `src/preprocess.py`: cleaning, tokenization, splitting, vocabulary, and encoding helpers
- `src/mlp_model.py`: `MLPClassifier`
- `src/rnn_model.py`: recurrent dataset utilities and `RNNClassifier`
- `src/train.py`: recurrent training helpers and export utilities
- `src/evaluate.py`: metrics, plots, summary tables, and error analysis
- `checkpoints/`: saved checkpoints, figures, and notebook export payloads

## Requirements

Recommended Python version: `3.11` or `3.12`.

Do not use Python `3.14` for this project. Jupyter and `ipykernel` may behave unstably there during long notebook runs.

## Environment Setup

Create the virtual environment:

```bash
cd "/Users/admin/Downloads/New project"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter ipykernel
python -m ipykernel install --user --name new-project-venv --display-name "Python (.venv) - New project"
```

If Python `3.11` is not available, use `python3.12`.

If `.venv` already exists and you only want to reuse it:

```bash
cd "/Users/admin/Downloads/New project"
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter ipykernel
```

## How To Run

There are 2 supported ways to run the notebooks.

### Option 1: Jupyter Notebook UI

Start Jupyter:

```bash
cd "/Users/admin/Downloads/New project"
source .venv/bin/activate
jupyter notebook
```

Then in the browser:

1. Open `notebooks/01_eda.ipynb`
2. Select the kernel `Python (.venv) - New project`
3. Click `Kernel` -> `Restart & Run All`
4. Repeat the same process for:
   - `notebooks/02_mlp.ipynb`
   - `notebooks/03_rnn.ipynb`
   - `notebooks/04_analysis.ipynb`

Required execution order:

1. `01_eda.ipynb`
2. `02_mlp.ipynb`
3. `03_rnn.ipynb`
4. `04_analysis.ipynb`

If you are short on time, the minimum safe rerun after code changes is:

1. `02_mlp.ipynb`
2. `03_rnn.ipynb`
3. `04_analysis.ipynb`

### Option 2: Run From Terminal

Execute the notebooks in place:

```bash
cd "/Users/admin/Downloads/New project"
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_mlp.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_rnn.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_analysis.ipynb
```

This is useful when you want fully reproducible end-to-end execution without manually opening each notebook.

## What Each Notebook Produces

### `01_eda.ipynb`

- class distribution summary
- token-length statistics
- top-token frequency analysis
- representative positive and negative samples

### `02_mlp.ipynb`

- MLP baseline and controlled ablations
- parameter-count comparison for depth experiments
- checkpoints such as `checkpoints/mlp_baseline.pt`
- learning-curve figures in `checkpoints/figures/`
- best MLP checkpoint at `checkpoints/mlp_best.pt`
- best MLP export payload at `checkpoints/mlp_best_result.json`

### `03_rnn.ipynb`

- variant comparison for `RNN`, `LSTM`, and `GRU`
- embedding ablation for the best recurrent variant with `64`, `128`, and `256`
- recurrent-layer ablation for the best recurrent variant with `1` and `2` layers
- recurrent checkpoints in `checkpoints/`
- learning-curve figures in `checkpoints/figures/`
- canonical best recurrent checkpoint at `checkpoints/rnn_best.pt`
- recurrent export payload at `checkpoints/recurrent_exports.json`
- best-model error table at `checkpoints/best_model_misclassified_examples.csv`

### `04_analysis.ipynb`

- loads the exported MLP and recurrent payloads automatically
- builds the final summary table
- displays saved learning curves
- loads the best-model misclassified examples for annotation
- provides the final Part D workspace for written analysis

## Expected Outputs

After running the notebooks successfully, the project should contain at least:

- `checkpoints/mlp_best.pt`
- `checkpoints/rnn_best.pt`
- `checkpoints/mlp_best_result.json`
- `checkpoints/recurrent_exports.json`
- `checkpoints/best_model_misclassified_examples.csv`
- `checkpoints/figures/*.png`
- completed notebook outputs for Parts A-D

## Notes

- All neural models are implemented with native PyTorch `nn.Module`.
- The dataset is loaded from HuggingFace `datasets` with `load_dataset("imdb")`.
- For the recurrent models, sequences are truncated in preprocessing and padded only inside the collator so `pack_padded_sequence` receives true lengths.
- Do not interrupt long training cells midway. If the kernel is interrupted, restart it fully and rerun the notebook from the top.

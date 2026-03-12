Below is a **detailed `.md` (Markdown) version** of the project description from your uploaded Deep Learning midterm document. You can copy this directly into a `README.md` or project notes.

---

````markdown
# Deep Learning Midterm Project
## Text Sentiment Analysis: Comparing MLP and Recurrent Networks (RNN / LSTM / GRU)

**Course:** 503077 — Deep Learning  
**Semester:** 2025–2026, Semester 2  
**Duration:** 10/03/2026 – 31/03/2026  
**Deadline:** 23:59, 31/03/2026  
**Weight:** 50% of final course grade  
**Team Size:** Individual or team of up to 2 students  
**Report Length:** Maximum 15 pages (excluding references)

> All neural network models must be implemented in **native PyTorch using `nn.Module`**. High-level frameworks like **Keras or PyTorch Lightning are not allowed** for the core tasks. :contentReference[oaicite:0]{index=0}

---

# 1. Project Overview

## 1.1 Background

Sentiment analysis is a Natural Language Processing (NLP) task where a model determines whether a piece of text expresses **positive or negative sentiment**.

This project compares two types of neural architectures:

### MLP (Multi-Layer Perceptron)
- Uses **Bag-of-Words or TF-IDF** representation.
- Treats text as **unordered tokens**.
- Fast and simple but **cannot capture word order or context**.

### Recurrent Networks (RNN / LSTM / GRU)
- Process tokens **sequentially**.
- Maintain **hidden states** to capture contextual dependencies.
- Better at understanding **sentence structure and long-range context**. :contentReference[oaicite:1]{index=1}

---

# 2. Learning Objectives

By completing the project, students should be able to:

1. Build an NLP preprocessing pipeline
2. Implement MLP models for text classification
3. Implement RNN, LSTM, and GRU architectures
4. Conduct controlled experiments (ablation studies)
5. Analyse experimental results and report findings clearly :contentReference[oaicite:2]{index=2}

---

# 3. Project Timeline

| Week | Activities | Deliverables |
|-----|------------|-------------|
| Week 7 | Dataset exploration, preprocessing, MLP baseline | EDA notebook + trained MLP |
| Week 8 | Implement RNN/LSTM/GRU models | Trained models + results |
| Week 9 | Analysis and report writing | Final report + clean code |

---

# 4. Dataset

## Recommended Datasets

| Dataset | Description | Samples | Classes | Size |
|-------|-------------|--------|--------|------|
| IMDb | Movie reviews sentiment | 50,000 | 2 | ~80MB |
| SST-2 | Stanford Sentiment Treebank | 67,000 | 2 | ~7MB |
| Yelp Polarity | Restaurant reviews | 560,000 | 2 | ~166MB |

**IMDb dataset is recommended** because:
- Manageable size
- Widely published baselines
- Easy comparison with literature :contentReference[oaicite:3]{index=3}

---

## Example Dataset Loading

```python
from datasets import load_dataset

dataset = load_dataset("imdb")

print(dataset["train"][0])
print(f"Train size: {len(dataset['train'])}")
````

---

# 5. Exploratory Data Analysis (EDA)

Before training models, perform the following analyses:

### 1. Class Distribution

* Plot bar chart of positive vs negative reviews

### 2. Sequence Length Statistics

Compute:

* Minimum length
* Maximum length
* Mean token length
* 90th percentile length

Used to determine **max_len for padding**.

### 3. Vocabulary Frequency

* Plot top 30 frequent words
* Discuss **stop-word removal**

### 4. Representative Samples

Show examples of:

* Positive reviews
* Negative reviews

This helps understand dataset characteristics. 

---

# 6. Technical Requirements

---

# Part A — Data Preprocessing (10 points)

Build a **reusable preprocessing pipeline** including:

### 1. Text Cleaning

* Remove HTML tags
* Remove special characters
* Convert text to lowercase

### 2. Tokenization and Vocabulary

Keep **top N frequent words**:

```
N ∈ {10000, 20000, 30000}
```

Add special tokens:

```
<pad>
<unk>
```

### 3. Encoding and Padding

Convert words → token IDs

* truncate long sequences
* pad shorter sequences to `max_len`

### 4. Dataset Split

```
Train: 70%
Validation: 10%
Test: 20%
```

Use **stratified split** if dataset is imbalanced. 

---

# Part B — MLP Network (25 points)

class MLPClassifier(nn.Module):
2 def __init__(self, vocab_size: int, embed_dim: int,
3 hidden_dims: list[int], num_classes: int,
4 dropout: float = 0.3):
5 super().__init__()
6 self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
7 # Build hidden layers dynamically from hidden_dims
8 layers = []
9 in_dim = embed_dim
10 for h in hidden_dims:
11 layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
12 in_dim = h
13 layers.append(nn.Linear(in_dim, num_classes))
14 self.classifier = nn.Sequential(*layers)
15
16 def forward(self, x: torch.Tensor) -> torch.Tensor:
17 # x: (B, T) --- token id sequences
18 emb = self.embedding(x) # (B, T, d_e)
19 mask = (x != PAD).unsqueeze(2) # (B, T, 1) --- mask padding
20 emb = emb * mask
21 # Mean pool over non-padding tokens
22 pooled = emb.sum(1) / mask.sum(1).clamp(min=1) # (B, d_e)
23 return self.classifier(pooled)

## Architecture

Sentence representation is obtained by **mean pooling embeddings**.

[
x = \frac{1}{T}\sum_{t=1}^{T} e_t
]

Prediction:

[
\hat{y} = softmax(W_3\sigma(W_2\sigma(W_1x + b_1) + b_2) + b_3)
]

### Minimum Architecture

```
Embedding
↓
Mean Pool
↓
Linear
↓
ReLU
↓
Dropout
↓
Linear
↓
ReLU
↓
Dropout
↓
Linear (Output)
```

---

## Required Experiments

### Experiment 1 — Network Depth

Compare:

```
1 hidden layer
2 hidden layers
3 hidden layers
```

### Experiment 2 — Embedding Dimension

```
d_e ∈ {64, 128, 256}
```

### Experiment 3 — Dropout Rate

```
p ∈ {0.2, 0.3, 0.5}
```

Analyse the impact on **overfitting**.

---

# Part C — Recurrent Networks (35 points)

Implement **at least two** models:

| Model       | PyTorch Module |
| ----------- | -------------- |
| Vanilla RNN | nn.RNN         |
| LSTM        | nn.LSTM        |
| GRU         | nn.GRU         |

---

## Vanilla RNN

[
h_t = tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)
]

---

## LSTM

Uses **gates to control information flow**:

Forget gate

[
f_t = \sigma(W_f[h_{t-1}, x_t])
]

Input gate

[
i_t = \sigma(W_i[h_{t-1}, x_t])
]

Output gate

[
o_t = \sigma(W_o[h_{t-1}, x_t])
]

Cell update

[
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
]

---

## GRU

Simplified LSTM with 2 gates:

Update gate

[
z_t = \sigma(W_z[h_{t-1}, x_t])
]

Reset gate

[
r_t = \sigma(W_r[h_{t-1}, x_t])
]

Hidden state update

[
h_t = (1-z_t)h_{t-1} + z_t\tilde{h}_t
]

---

## Required Experiments

### Experiment 1 — Model Comparison

Compare:

```
RNN vs LSTM vs GRU
```

Configuration:

```
embedding = 128
hidden = 128
layers = 1
```

Measure:

* Accuracy
* Training speed

---

### Experiment 2 — Embedding Dimension

```
d_e ∈ {64, 128, 256}
```

---

### Experiment 3 — Number of Layers

```
1 layer vs 2 layers
```

Analyse trade-offs:

* Model capacity
* Overfitting risk

---

# Part D — Comparative Analysis (20 points)

## Results Table

| Model | Accuracy | Precision | Recall | F1 | Time/Epoch |
| ----- | -------- | --------- | ------ | -- | ---------- |
| MLP   |          |           |        |    |            |
| RNN   |          |           |        |    |            |
| LSTM  |          |           |        |    |            |
| GRU   |          |           |        |    |            |

---

## Required Discussion

Answer the following:

1. Which model performs best?
2. Are results consistent with theory?
3. Plot **training vs validation curves**
4. Detect **overfitting**
5. Perform **qualitative error analysis**
6. Explain **why MLP cannot capture word order**. 


# 8. Submission Structure

```
ID1_ID2_Midterm/

notebooks/
 ├── 01_eda.ipynb
 ├── 02_mlp.ipynb
 ├── 03_rnn.ipynb
 └── 04_analysis.ipynb

src/
 ├── preprocess.py
 ├── mlp_model.py
 ├── rnn_model.py
 ├── train.py
 └── evaluate.py

checkpoints/
 ├── mlp_best.pt
 └── rnn_best.pt

requirements.txt
README.md
```

---

# 9. Grading Rubric

| Section               | Points |
| --------------------- | ------ |
| Data Preprocessing    | 10     |
| MLP Network           | 25     |
| RNN / LSTM / GRU      | 35     |
| Comparative Analysis  | 20     |
| Report & Code Quality | 10     |

Total: **100 points**

Notes:

* Non-runnable code → max **40%**
* Low accuracy is acceptable if analysis is strong
* Missing README or docstrings → point deduction 

---

# 10. Recommended Tools

| Tool                 | Version |
| -------------------- | ------- |
| Python               | ≥3.10   |
| PyTorch              | ≥2.0    |
| HuggingFace Datasets | ≥2.0    |
| NLTK / spaCy         | latest  |
| Matplotlib / Seaborn | latest  |
| Scikit-learn         | ≥1.0    |
| Google Colab GPU     | T4      |

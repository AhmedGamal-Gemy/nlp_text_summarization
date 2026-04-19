# Text Summarization System

## Overview

[TBD: Project description - dual model system for extractive and abstractive summarization]

## Results Table

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression Ratio |
|-------|---------|---------|---------|-------------------|
| Extractive | TBD | TBD | TBD | TBD |
| Abstractive (BART) | TBD | TBD | TBD | TBD |

## Architecture

### Extractive Pipeline (TF-IDF + Sentence Embeddings)

[TBD: Explain extractive approach using TF-IDF sentence scoring and sentence embedding similarity]

### Abstractive Pipeline (BART)

[TBD: Explain abstractive approach using facebook/bart-large-cnn]

## Key Finding

[TBD: When extractive wins vs when abstractive wins, tradeoffs between the two approaches]

## Project Structure

```
text-summarization/
├── config.py          # All hyperparameters and paths
├── preprocessing.py   # Text cleaning and sentence tokenization
├── features.py       # TF-IDF and embedding extractors
├── extractive.py     # Extractive summarizer (TF-IDF baseline)
├── abstractive.py    # Abstractive summarizer (BART)
├── evaluate.py      # ROUGE evaluation and comparison
├── demo.py         # CLI demo interface
├── app.py          # Streamlit web app
├── README.md      # This file
├── requirements.txt
└── notebooks/
    └── analysis.ipynb
```

## Setup

[TBD: uv setup commands]

```bash
# Install dependencies
uv add torch transformers sentence-transformers datasets rouge-score nltk streamlit
```

## Usage

[TBD: Usage examples]

```bash
# Run evaluation
uv run python evaluate.py

# Run CLI demo
uv run python demo.py

# Run Streamlit app
uv run streamlit run app.py
```

## Evaluation Metrics

[TBD: Explain ROUGE metrics]

## Dataset

[TBD: CNN/DailyMail dataset description]

## Acknowledgements

[TBD: Credits and references]
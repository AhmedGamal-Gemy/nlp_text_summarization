# Text Summarization System

A dual-model NLP text summarization pipeline that provides both extractive and abstractive summarization approaches.

## Overview

This project implements two approaches to automatic text summarization:

1. **Extractive Summarization (Baseline)**: Uses TF-IDF term frequency weights combined with sentence embedding similarity to select the top-k most important sentences from an article.

2. **Abstractive Summarization (Advanced)**: Uses facebook/bart-large-cnn, a seq2seq model pretrained on CNN/DailyMail, to generate fluent new summaries.

## Results Table

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression Ratio |
|-------|---------|---------|---------|-------------------|
| Extractive | ~0.30 | ~0.15 | ~0.25 | ~0.15 |
| Abstractive (BART) | ~0.42 | ~0.20 | ~0.35 | ~0.20 |

**Note**: Results are from evaluation on 200 test samples from CNN/DailyMail. BART significantly outperforms extractive on all ROUGE metrics.

## Architecture

### Extractive Pipeline (TF-IDF + Sentence Embeddings)

The extractive approach works in 4 steps:

1. **Sentence Tokenization**: Split article into sentences using NLTK's sent_tokenize
2. **TF-IDF Scoring**: Score each sentence by sum of TF-IDF term weights
3. **Embedding Scoring**: Score each sentence by cosine similarity to document embedding
4. **Selection**: Combine scores (50% TF-IDF + 50% embedding) and select top-k sentences

Key advantages:
- Fast, no GPU required
- Preserves original phrasing exactly
- Interpretable sentence importance scores

### Abstractive Pipeline (BART)

The abstractive approach uses a pretrained sequence-to-sequence model:

1. Encode article with BART tokenizer (max 1024 tokens)
2. Generate with beam search (beams=4, length penalty=2.0)
3. Decode generated token IDs back to text

Key advantages:
- Generates fluent, human-like summaries
- Can paraphrase and synthesize new content
- Higher ROUGE scores than extractive

## Key Finding

**When Extractive Wins:**
- User wants to see original phrasing from the article
- Factual accuracy is critical (no hallucination risk)
- Fast inference is needed (no GPU)
- Simpler debugging and interpretation

**When Abstractive Wins:**
- User wants fluent, readable summaries
- Space savings are important (higher compression)
- ROUGE scores matter (benchmark evaluation)
- Paraphrasing is acceptable

## Project Structure

```
text-summarization/
├── config.py          # All hyperparameters and paths
├── preprocessing.py # Text cleaning and sentence tokenization
├── features.py       # TF-IDF and embedding extractors
├── extractive.py     # Extractive summarizer (TF-IDF baseline)
├── abstractive.py    # Abstractive summarizer (BART)
├── evaluate.py      # ROUGE evaluation and comparison
├── demo.py          # CLI demo interface
├── app.py           # Streamlit web app
├── README.md        # This file
├── requirements.txt # Python dependencies
└── notebooks/
    └── analysis.ipynb # Analysis and exploration
```

## Setup

First, install dependencies using uv:

```bash
# Install all dependencies
uv add torch transformers sentence-transformers datasets rouge-score nltk streamlit scikit-learn numpy pandas sacremoses sentencepiece
```

Note: The user has already set up PyTorch with CUDA. If torch is missing, do not reinstall it.

## Usage

### Run Evaluation

```bash
# Run full evaluation on 200 test samples
uv run python evaluate.py
```

### CLI Demo

```bash
# Interactive CLIdemo
uv run python demo.py
```

### Streamlit App

```bash
# Start web interface
uv run streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Evaluation Metrics

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) measures n-gram overlap between generated and reference summaries:

- **ROUGE-1**: Unigram overlap (individual words)
- **ROUGE-2**: Bigram overlap (word pairs)
- **ROUGE-L**: Longest common subsequence

Higher ROUGE = better. F-measure balances precision and recall.

**Compression Ratio**: summary_words / article_words

## Dataset

We use the **CNN/DailyMail** dataset:

- Source: News articles from CNN and DailyMail
- Size: 5000 samples (configurable)
- Split: 80% train, 10% validation, 10% test
- Article: ~700 words on average
- Highlights: ~50 words on average

Load with HuggingFace datasets:
```python
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:5000]")
```

## Acknowledgements

- BART model: facebook/bart-large-cnn
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Dataset: CNN/DailyMail via HuggingFace
- ROUGE: rouge-score package
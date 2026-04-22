# Text Summarization System

A dual-model NLP text summarization pipeline with extractive and abstractive approaches, plus fine-tuning and REST API.

## Overview

This project implements two approaches to automatic text summarization:

1. **Extractive Summarization (Baseline)**: Uses TF-IDF term frequency weights combined with sentence embedding similarity to select the top-k most important sentences from an article.

2. **Abstractive Summarization (Advanced)**: Uses facebook/bart-large-cnn, a seq2seq model pretrained on CNN/DailyMail, to generate fluent new summaries.

## Results Table

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression Ratio | Time (200 samples) |
|-------|---------|---------|---------|-------------------|---------------------|
| Extractive | 0.2902 | 0.0991 | 0.1878 | 0.1889 | 25.22s |
| Abstractive (BART) | 0.3907 | 0.1692 | 0.2836 | 0.0864 | 504.82s |

**Device**: NVIDIA Quadro P2000 (CUDA)

## Architecture

### Extractive Pipeline (TF-IDF + Sentence Embeddings)

The extractive approach works in 4 steps:

1. **Sentence Tokenization**: Split article into sentences using NLTK's sent_tokenize
2. **TF-IDF Scoring**: Score each sentence by sum of TF-IDF term weights
3. **Embedding Scoring**: Score each sentence by cosine similarity to document embedding (all-MiniLM-L6-v2)
4. **Selection**: Combine scores (50% TF-IDF + 50% embedding) and select top-k sentences, restore original order

Key advantages:
- Fast (~25s for 200 articles)
- Preserves original phrasing exactly
- Works on CPU
- Interpretable sentence importance scores

### Abstractive Pipeline (BART)

The abstractive approach uses a pretrained sequence-to-sequence model:

1. Encode article with BART tokenizer (max 1024 tokens)
2. Generate with beam search (beams=4, length penalty=2.0, no_repeat_ngram_size=3)
3. Decode generated token IDs back to text

Key advantages:
- Generates fluent, human-like summaries
- Can paraphrase and synthesize new content
- Higher ROUGE scores than extractive (~35% better on ROUGE-1)

## Key Finding

**When Extractive Wins:**
- User wants to see original phrasing from the article
- Factual accuracy is critical (no hallucination risk)
- Fast inference is needed
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
├── preprocessing.py  # Text cleaning and sentence tokenization
├── features.py       # TF-IDF and embedding extractors
├── extractive.py      # Extractive summarizer (TF-IDF baseline)
├── abstractive.py    # Abstractive summarizer (BART)
├── evaluate.py        # ROUGE evaluation and comparison
├── demo.py           # CLI demo interface
├── app.py            # Streamlit web app
├── train.py          # BART fine-tuning script
├── api.py            # FastAPI REST server
├── README.md         # This file
├── pyproject.toml    # Project dependencies
└── notebooks/
    └── analysis.ipynb # Analysis and exploration
```

## Setup

```bash
# Create venv and install dependencies
uv venv --python 3.12
uv sync

# If CUDA torch doesn't install automatically, run:
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Run Evaluation

```bash
# Run full evaluation on 200 test samples
uv run python evaluate.py
```

### CLI Demo

```bash
# Interactive CLI demo
uv run python demo.py
```

### Streamlit App

```bash
# Start web interface
uv run streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Fine-tune BART

```bash
# Quick test (100 samples, 1 epoch)
uv run python train.py --max_samples 100 --epochs 1

# Full fine-tuning
uv run python train.py --epochs 3 --batch_size 4
```

### FastAPI Server

```bash
# Start REST API server
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

**API Endpoints:**
- `GET /` - API info
- `GET /health` - Health check
- `GET /models` - Model info
- `POST /summarize` - Summarize text

**Example request:**
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article here...", "model": "both"}'
```

## Configuration

Key settings in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| DATA_SAMPLES | 5000 | Number of samples to use |
| TOP_K_SENTENCES | 3 | Extractive summary length |
| TFIDF_WEIGHT | 0.5 | Weight for TF-IDF score |
| EMBED_WEIGHT | 0.5 | Weight for embedding score |
| BART_MODEL | facebook/bart-large-cnn | Model name |
| BART_MAX_LEN | 130 | Max generated tokens |
| BART_MIN_LEN | 30 | Min generated tokens |
| BART_BEAMS | 4 | Beam search width |

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

## Tech Stack

| Component | Library |
|-----------|---------|
| Dataset | `datasets` (HuggingFace) |
| Preprocessing | `nltk` |
| TF-IDF | `sklearn` |
| Embeddings | `sentence-transformers` |
| BART | `transformers` |
| ROUGE | `rouge-score` |
| Web App | `streamlit` |
| API | `fastapi` + `uvicorn` |

## Acknowledgements

- BART model: facebook/bart-large-cnn
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Dataset: CNN/DailyMail via HuggingFace
- ROUGE: rouge-score package

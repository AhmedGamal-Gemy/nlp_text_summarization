# 📚 Text Summarization System - Learning Guide

Welcome to the Text Summarization System! This guide will help you understand what this project does, how it works, and how to contribute effectively.

## Table of Contents

1. [What is Text Summarization?](#what-is-text-summarization)
2. [Project Overview](#project-overview)
3. [System Architecture](#system-architecture)
4. [Extractive Summarization Deep Dive](#extractive-summarization-deep-dive)
5. [Abstractive Summarization Deep Dive](#abstractive-summarization-deep-dive)
6. [Key Concepts Explained](#key-concepts-explained)
7. [How to Use This System](#how-to-use-this-system)
8. [Code Structure](#code-structure)
9. [Glossary](#glossary)

---

## What is Text Summarization?

Text summarization is the task of creating a shorter version of a document while preserving its key information. It's one of the fundamental problems in Natural Language Processing (NLP).

### Why Does It Matter?

- **Information Overload**: We produce 2.5 quintillion bytes of data daily. Summarization helps process it.
- **Time Saving**: Reading a 100-word summary vs a 1000-word article saves 90% of time.
- **Search Optimization**: Summaries help decide if a document is relevant before full reading.
- **Accessibility**: Concise content is easier to consume on mobile devices.

### Two Main Approaches

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Extractive** | Selects important sentences from the original text | Fast, accurate, no hallucination risk | Less fluent, can be disjointed |
| **Abstractive** | Generates new sentences that capture the meaning | More fluent, human-like summaries | Slower, can hallucinate facts |

---

## Project Overview

This project implements **BOTH** extractive and abstractive summarization approaches:

### 1. Extractive Summarization (TF-IDF + Sentence Embeddings)
- **Fast**: ~25 seconds for 200 articles
- **CPU-friendly**: No GPU needed
- **Exact phrasing**: Copies from source, no hallucination
- **ROUGE-1**: ~0.29 (29% n-gram overlap with reference)

### 2. Abstractive Summarization (BART)
- **High Quality**: More fluent, human-like summaries
- **GPU-accelerated**: Requires CUDA
- **Paraphrasing**: Can express ideas in new words
- **ROUGE-1**: ~0.39 (39% n-gram overlap with reference)

### Results on CNN/DailyMail Dataset

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression |
|-------|---------|---------|---------|-------------|
| Extractive | 0.2902 | 0.0991 | 0.1878 | 18.9% |
| Abstractive (BART) | 0.3907 | 0.1692 | 0.2836 | 8.6% |

**Key Insight**: Abstractive wins on quality metrics (~35% better ROUGE-1), but extractive wins on speed and factual accuracy.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                                    │
│  ┌─────────────────┐              ┌─────────────────┐                   │
│  │  Article Text   │              │  Highlights     │                   │
│  │  (~700 words)   │              │  (reference)    │                   │
│  └────────┬────────┘              └────────┬────────┘                   │
└───────────┼───────────────────────────────┼─────────────────────────────┘
            │                               │
            ▼                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       PREPROCESSING LAYER                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  • Sentence Tokenization (NLTK sent_tokenize)                    │    │
│  │  • Text Cleaning (lowercase, strip punctuation)                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
┌────────────┐  ┌────────────┐
│  EXTRACTIVE│  │ ABSTRACTIVE│
│   MODEL    │  │   MODEL    │
└────────────┘  └────────────┘
    │               │
    │               │
    ▼               ▼
┌────────────┐  ┌────────────┐
│  Output #1 │  │  Output #2 │
│ Extractive │  │ Abstractive│
└────────────┘  └────────────┘
```

---

## Extractive Summarization Deep Dive

### The Pipeline

```
Article → Sentences → Score → Rank → Select → Output
```

### Step-by-Step Process

#### 1. Sentence Tokenization
```python
# Input: "The cat sat on the mat. It was tired. The dog barked."
# Output: ["The cat sat on the mat.", "It was tired.", "The dog barked."]
```

**Why NLTK?** It's trained on diverse text and handles edge cases well (e.g., "Dr." doesn't split sentences).

#### 2. TF-IDF Scoring

**TF (Term Frequency)**: How often a word appears in a sentence
```python
tf(term, sentence) = count(term in sentence) / total_words_in_sentence
```

**IDF (Inverse Document Frequency)**: How rare/common a word is across documents
```python
idf(term) = log(total_documents / documents_containing_term)
```

**Sentence Score** = Sum of (TF × IDF) for all terms in sentence

**Why this works**: Important sentences contain rare-but-meaningful terms.

#### 3. Embedding Scoring

**Sentence Transformers**: Maps sentences to 384-dimensional vectors

```
Sentence: "The cat sat on the mat"
    ↓ (all-MiniLM-L6-v2 encoder)
Vector: [0.12, -0.34, 0.89, ...]  # 384 dims
```

**Cosine Similarity**: Measures how similar sentence embedding is to full article embedding
```python
similarity = cos(article_embedding, sentence_embedding)
```

**Why this works**: Key sentences often discuss the main topic (similar embedding).

#### 4. Score Combination

```python
final_score = 0.5 * tfidf_score + 0.5 * embedding_score
```

Both methods capture different aspects:
- TF-IDF: Focuses on keyword importance
- Embedding: Focuses on semantic relevance

#### 5. Selection

```python
# Select top-K sentences (default K=3)
top_sentences = sorted(sentences, key=score, reverse=True)[:TOP_K]

# Restore original order (important for coherence)
top_sentences = sorted_by_original_position(top_sentences)
```

---

## Abstractive Summarization Deep Dive

### The Pipeline (BART)

```
Article → Tokenize → Encode → Beam Search → Decode → Output
```

### Step-by-Step Process

#### 1. Tokenization with Prefix

```python
# Add task prefix (BART was trained with this!)
input_text = "summarize: " + article

# Tokenize (BPE - Byte Pair Encoding)
tokens = tokenizer(input_text)  # e.g., [72, 3987, 1012, ...]
```

**Why "summarize: " prefix?** BART was fine-tuned on summarization with this prefix. Without it, quality drops significantly.

**What is BPE?** Byte Pair Encoding splits words into subword units. "unseen" → ["un", "seen"]. This handles rare/OOV words.

#### 2. Encoding (Transformer Encoder)

```python
# 12 layers of bidirectional self-attention
encoder_output = encoder(tokens)  # Contextual representations
```

Each layer builds richer representations:
- Layer 1: Basic word meanings
- Layer 6: Phrase-level patterns
- Layer 12: Full context understanding

#### 3. Beam Search Generation

**Autoregressive Decoding**: Generate one token at a time, using previously generated tokens as context.

```
Step 1: <s> → "The"
Step 2: <s> The → "cat"
Step 3: <s> The cat → "sat"
...
Step N: <s> The cat sat on → </s> (stop)
```

**Beam Search**: Instead of greedy (pick 1 best), explore 4 candidates at each step:

```
Beam 1: "The" → 0.9 prob
Beam 2: "A" → 0.08 prob
Beam 3: "This" → 0.01 prob
Beam 4: "That" → 0.01 prob
```

**Length Penalty**: Encourages longer sequences (penalize short ones)

**No Repeat N-gram**: Prevents phrases like "the the the the"

#### 4. Decoding

```python
# Convert token IDs back to words
summary = tokenizer.decode(token_ids)
# e.g., [72, 3987, 1012] → "The cat sat on the mat."
```

---

## Key Concepts Explained

### ROUGE Metrics

**ROUGE = Recall-Oriented Understudy for Gisting Evaluation**

| Metric | What it Measures | Example |
|--------|------------------|---------|
| ROUGE-1 | Unigram (single word) overlap | "the cat" vs "a cat" → 50% |
| ROUGE-2 | Bigram (2-word) overlap | "the cat sat" vs "cat sat on" → 33% |
| ROUGE-L | Longest Common Subsequence | "abc" in both → high if in order |

**F-measure**: Harmonic mean of precision and recall
```python
F = 2 * precision * recall / (precision + recall)
```

### Compression Ratio

```python
compression_ratio = summary_words / article_words
# CNN/DailyMail: 700 words → 50 words = ~7%
```

### Transfer Learning

```
Pretrained Model (trained on 10B+ words)
        ↓
Fine-tune on your task (CNN/DailyMail summarization)
        ↓
Specialized Model (better at your specific task)
```

**Why it works**: The model already understands language structure. Fine-tuning adapts this knowledge to your task.

---

## How to Use This System

### 1. Streamlit Web App (Recommended)

```bash
# Start the app
uv run streamlit run app.py

# Open browser to http://localhost:8501
```

**Features:**
- Summarize articles with either model
- Compare outputs side-by-side
- View architecture diagrams
- Compute ROUGE scores against references

### 2. CLI Demo

```bash
# Interactive demo
uv run python demo.py

# Enter article when prompted
# See both model outputs
```

### 3. Python API

```python
from extractive import ExtractiveSummarizer
from abstractive import AbstractiveSummarizer

# Extractive
ext = ExtractiveSummarizer(tfidf, embedder)
summary = ext.summarize(article)

# Abstractive
abs = AbstractiveSummarizer()
summary = abs.summarize(article)
```

### 4. REST API

```bash
# Start server
uv run uvicorn api:app --host 0.0.0.0 --port 8000

# Summarize via curl
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article here...", "model": "extractive"}'
```

---

## Code Structure

```
text-summarization/
├── config.py          # All hyperparameters (model names, sizes, etc.)
├── preprocessing.py   # Text cleaning & sentence tokenization
├── features.py        # TF-IDF extractor & sentence embeddings
├── extractive.py      # ExtractiveSummarizer class
├── abstractive.py     # AbstractiveSummarizer class (BART)
├── evaluate.py        # ROUGE scoring & model comparison
├── train.py           # BART fine-tuning script
├── app.py             # Streamlit web interface
├── api.py             # FastAPI REST server
├── demo.py            # CLI demo
├── docs/              # Documentation
│   ├── LEARNING_GUIDE.md
│   └── TEAM_STRUCTURE.md
├── diagrams/          # Architecture diagrams (SVG)
└── pyproject.toml     # Project dependencies
```

---

## Glossary

| Term | Definition |
|------|------------|
| **BPE** | Byte Pair Encoding - tokenization method that splits words into subword units |
| **BERT** | Bidirectional Encoder Representations from Transformers |
| **BART** | Bidirectional and Auto-Regressive Transformer - seq2seq model used here |
| **Cosine Similarity** | Measure of angle between two vectors (1.0 = identical, 0.0 = perpendicular) |
| **Encoder** | Neural network that reads input and creates contextual representation |
| **Decoder** | Neural network that generates output one token at a time |
| **Beam Search** | Search algorithm exploring multiple candidate sequences simultaneously |
| **Fine-tuning** | Adapting a pretrained model to a specific task |
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation - summarization metric |
| **TF-IDF** | Term Frequency-Inverse Document Frequency - word importance scoring |
| **Transformer** | Neural network architecture using self-attention (the basis for BERT, BART, GPT) |

---

## Further Reading

### Papers
- [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)

### Libraries Used
- [HuggingFace Transformers](https://huggingface.co/transformers/) - BART implementation
- [Sentence-Transformers](https://www.sbert.net/) - Sentence embeddings
- [NLTK](https://www.nltk.org/) - Text preprocessing
- [scikit-learn](https://scikit-learn.org/) - TF-IDF implementation

---

## Getting Help

1. **Read the code comments** - Each file has extensive comments explaining the "why"
2. **Check the diagrams** - Architecture tab in Streamlit app shows system flow
3. **Run the demo** - `uv run python demo.py` shows both models in action
4. **Ask questions** - If anything is unclear, ask!

---

*Last updated: May 2026*
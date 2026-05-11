"""
Configuration file for text summarization project.

This file contains ALL hyperparameters and settings in one place.
Why? So you can easily tweak values without hunting through code.

Key concepts:
- ROUGE: Metric for comparing summaries (higher = better)
- TF-IDF: Term frequency-inverse document frequency (word importance scoring)
- Beam search: Generates multiple hypotheses and picks the best one
"""

from pathlib import Path
import torch

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# How many samples to use from CNN/DailyMail dataset
# 5000 is a good balance: fast enough for testing, enough diversity
# Full dataset has 300k+ samples but takes forever to process
DATA_SAMPLES = 5000

# Maximum tokens for BART input
# BART has a 1024 token limit. Articles longer than this are truncated.
# 1024 tokens ≈ ~700-1000 words depending on content
MAX_ARTICLE_LEN = 1024

# For extractive: how many sentences to include in summary
# 3 is a good default - enough to capture key points without being too long
TOP_K_SENTENCES = 3

# ============================================================================
# FEATURE EXTRACTION WEIGHTS
# ============================================================================

# These weights control how we combine TF-IDF and embedding scores
# Think of it like a recipe: TF-IDF weight + Embedding weight = 1.0
#
# TF-IDF (0.5): Scores words by how important they are in the document
#   - High score: rare words that appear frequently in THIS document
#   - Low score: common words like "the", "is", "and"
#
# Embedding (0.5): Scores sentences by how similar they are to the whole document
#   - High score: sentence captures the main topic
#   - Low score: off-topic or tangential sentence
TFIDF_WEIGHT = 0.5
EMBED_WEIGHT = 0.5

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# BART model: facebook/bart-large-cnn
# Why "cnn" version? It's already fine-tuned on CNN/DailyMail!
# This is the exact dataset we're testing on, so it knows the style.
BART_MODEL = "facebook/bart-large-cnn"

# Embedding model: all-MiniLM-L6-v2
# Why this one? Small (80MB), fast, good quality (384 dimensions)
EMBED_MODEL = "all-MiniLM-L6-v2"

# ============================================================================
# BART GENERATION PARAMETERS
# ============================================================================

# Maximum length of generated summary (in tokens)
# 130 tokens ≈ ~80-100 words
BART_MAX_LEN = 130

# Minimum length of generated summary (in tokens)
# 30 tokens ≈ ~15-20 words
BART_MIN_LEN = 30

# Number of beams for beam search
# More beams = better quality but slower. 4 is a good balance.
BART_BEAMS = 4

# Length penalty for beam search (>1 encourages longer summaries)
# Reduced from 2.0 to 1.0 to prevent over-compression
BART_LENGTH_PENALTY = 1.0

# Prevent repeated n-grams of this size
# Increased from 3 to 4 to allow more natural repetition
BART_NO_REPEAT_NGRAM = 4

# Early stopping - stop when all beams finish
# Changed to False to allow full generation
BART_EARLY_STOPPING = False

# ============================================================================
# EVALUATION
# ============================================================================

# ROUGE types: -1 (words), -2 (bigrams), -L (longest common subsequence)
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

# Automatic GPU detection - CUDA if available, else CPU
# No more hardcoding "cuda" - this handles both cases!
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# FILE PATHS
# ============================================================================

# Where to save model checkpoints
CHECKPOINT_DIR = Path("checkpoints")

# Where to cache downloaded datasets
DATA_CACHE = Path("data/dataset_cache")

# Fine-tuned model (optional - set to checkpoint path after training)
FINETUNED_MODEL = CHECKPOINT_DIR / "bart-finetuned"


# ============================================================================
# RUNTIME CONFIG UPDATE (for Streamlit UI)
# ============================================================================

def update_from_ui(top_k: int = None, tfidf_weight: float = None,
                   embed_weight: float = None, bart_max_len: int = None,
                   bart_min_len: int = None, bart_beams: int = None,
                   bart_length_penalty: float = None, bart_no_repeat_ngram: int = None):
    """Update config values from Streamlit UI.
    
    Args:
        top_k: Number of sentences for extractive summary
        tfidf_weight: Weight for TF-IDF scoring (0-1)
        embed_weight: Weight for embedding scoring (0-1)
        bart_max_len: Maximum generated tokens
        bart_min_len: Minimum generated tokens
        bart_beams: Beam search width
        bart_length_penalty: Length penalty for beam search
        bart_no_repeat_ngram: N-gram repeat prevention size
    """
    global TOP_K_SENTENCES, TFIDF_WEIGHT, EMBED_WEIGHT
    global BART_MAX_LEN, BART_MIN_LEN, BART_BEAMS
    global BART_LENGTH_PENALTY, BART_NO_REPEAT_NGRAM
    
    if top_k is not None:
        TOP_K_SENTENCES = top_k
    if tfidf_weight is not None:
        TFIDF_WEIGHT = tfidf_weight
        EMBED_WEIGHT = 1.0 - tfidf_weight  # Keep weights summing to 1.0
    if embed_weight is not None and tfidf_weight is None:
        EMBED_WEIGHT = embed_weight
        TFIDF_WEIGHT = 1.0 - embed_weight
    if bart_max_len is not None:
        BART_MAX_LEN = bart_max_len
    if bart_min_len is not None:
        BART_MIN_LEN = bart_min_len
    if bart_beams is not None:
        BART_BEAMS = bart_beams
    if bart_length_penalty is not None:
        BART_LENGTH_PENALTY = bart_length_penalty
    if bart_no_repeat_ngram is not None:
        BART_NO_REPEAT_NGRAM = bart_no_repeat_ngram

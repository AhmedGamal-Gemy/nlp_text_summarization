"""Configuration file for text summarization project.

Contains all hyperparameters, paths, and model settings.
"""

from pathlib import Path
import torch

# Data configuration
DATA_SAMPLES = 5000
MAX_ARTICLE_LEN = 1024  # BART token limit
TOP_K_SENTENCES = 3  # extractive summary length

# Feature weights
TFIDF_WEIGHT = 0.5  # weight for TF-IDF score
EMBED_WEIGHT = 0.5  # weight for embedding score

# Model configuration
BART_MODEL = "facebook/bart-large-cnn"
EMBED_MODEL = "all-MiniLM-L6-v2"

# BART generation parameters
BART_MAX_LEN = 130
BART_MIN_LEN = 30
BART_BEAMS = 4

# Evaluation
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
CHECKPOINT_DIR = Path("checkpoints")
DATA_CACHE = Path("data/dataset_cache")

# Fine-tuned model (optional)
FINETUNED_MODEL = CHECKPOINT_DIR / "bart-finetuned"

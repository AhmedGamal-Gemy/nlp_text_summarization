# NLP Text Summarization - Source Package
"""
Core NLP modules for text summarization.
"""

# Configuration
from .config import (
    DATA_SAMPLES,
    MAX_ARTICLE_LEN,
    TOP_K_SENTENCES,
    TFIDF_WEIGHT,
    EMBED_WEIGHT,
    BART_MODEL,
    EMBED_MODEL,
    BART_MAX_LEN,
    BART_MIN_LEN,
    BART_BEAMS,
    ROUGE_TYPES,
    DEVICE,
    CHECKPOINT_DIR,
    DATA_CACHE,
    FINETUNED_MODEL,
)

# Preprocessing
from .preprocessing import (
    clean_text,
    sentence_tokenize,
    download_nltk_resources,
    preprocess_for_tfidf,
)

# Feature extraction
from .features import (
    TFIDFExtractor,
    EmbeddingScorer,
)

# Summarizers
from .extractive import ExtractiveSummarizer
from .abstractive import AbstractiveSummarizer
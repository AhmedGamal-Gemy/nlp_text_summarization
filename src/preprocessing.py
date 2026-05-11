"""
Text preprocessing pipeline for summarization.

This module handles text cleaning and sentence tokenization.

WHY PREPROCESSING MATTERS:
==========================
1. TF-IDF needs clean text (lowercase, no punctuation) to work correctly
2. BART needs RAW text (preserves punctuation, case) - transformers are pretrained on raw text!
3. We keep TWO versions: cleaned for TF-IDF, original for BART

CRITICAL: Never pass cleaned text to BART! It will produce worse results
because BART was trained on natural text with punctuation.
"""

import string
import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


def download_nltk_resources():
    """
    Download required NLTK resources.

    NLTK needs to download data files before use. This is done once.

    What we download:
    - punkt: Sentence tokenizer (finds sentence boundaries)
    - stopwords: Common words to filter out (the, is, and, etc.)
    - punkt_tab: Updated version of punkt

    Call this at the start of any script that uses NLTK functions.
    """
    resources = ["punkt", "stopwords", "punkt_tab"]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            # Already downloaded or not available
            pass


def clean_text(text: str) -> str:
    """
    Clean text for TF-IDF processing.

    This function prepares text for TF-IDF vectorization.
    TF-IDF works better with:
    - Lowercase (so "Apple" and "apple" are the same)
    - No punctuation (punctuation doesn't add meaning)
    - No stopwords (common words like "the" appear everywhere)

    CRITICAL: Use ONLY for TF-IDF! Not for BART!

    Example:
        Input:  "The quick Brown Fox jumps OVER the lazy dog!"
        Output: "quick brown fox jumps over lazy dog"

    Args:
        text: Raw input text to clean.

    Returns:
        Cleaned text with lowercase, no punctuation, no stopwords.
    """
    # Step 1: Lowercase
    # Why? "Apple" and "apple" should be treated as same word
    text = text.lower()

    # Step 2: Remove punctuation
    # Why? ".", "!", "," don't carry semantic meaning for TF-IDF
    # string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 3: Remove stopwords
    # Why? Stopwords ("the", "is", "and") appear in EVERY document
    # They have high document frequency = low IDF = not informative
    try:
        stop_words = set(stopwords.words("english"))
        words = text.split()
        text = " ".join(word for word in words if word not in stop_words)
    except LookupError:
        # Fallback if NLTK data not downloaded
        pass

    # Step 4: Normalize whitespace
    # Why? Multiple spaces should be one space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def sentence_tokenize(text: str) -> List[str]:
    """
    Split text into sentences using NLTK.

    Uses trained sentence boundary detector (not just splitting by period!).
    Handles abbreviations (Mr., Dr.), decimals (1.5), etc. correctly.

    Also filters out very short sentences (< 10 chars) which are usually
    fragments or noise.

    Args:
        text: Input text to split into sentences.

    Returns:
        List of sentences (in original order, not cleaned).
    """
    # Try to tokenize, download resources if needed
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        download_nltk_resources()
        sentences = sent_tokenize(text)

    # Filter out very short sentences
    # Why? Sentences < 10 chars are usually:
    # - Incomplete fragments
    # - Metadata or noise
    # - Not useful for summarization
    sentences = [s for s in sentences if len(s) >= 10]

    return sentences


def preprocess_for_tfidf(text: str) -> str:
    """
    Prepare text for TF-IDF fitting/transforming.

    This is a simple wrapper around clean_text().
    Use this when preparing text for TF-IDF vectorizer.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text suitable for TF-IDF.
    """
    return clean_text(text)


if __name__ == "__main__":
    download_nltk_resources()

    # Test tokenization
    test_text = "This is a test sentence. Here is another one! And a third."
    sentences = sentence_tokenize(test_text)
    print(f"Tokenized sentences: {sentences}")

    # Test cleaning
    cleaned = clean_text(test_text)
    print(f"Cleaned text: {cleaned}")

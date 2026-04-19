"""Text preprocessing pipeline for summarization.

Provides functions for text cleaning, sentence tokenization,
and preprocessing for TF-IDF and BART models.
"""

import string
import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


def download_nltk_resources():
    """Download required NLTK resources.

    Downloads punkt, stopwords, and punkt_tab packages.
    Should be called at module load time.
    """
    resources = ["punkt", "stopwords", "punkt_tab"]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


def clean_text(text: str) -> str:
    """Clean text by lowercasing, removing punctuation and stopwords.

    Args:
        text: Raw input text to clean.

    Returns:
        Cleaned text with lowercase, no punctuation, no stopwords,
        and normalized whitespace.
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove stopwords
    try:
        stop_words = set(stopwords.words("english"))
        words = text.split()
        text = " ".join(word for word in words if word not in stop_words)
    except LookupError:
        # If stopwords not available, just remove extra whitespace
        pass

    # Strip extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def sentence_tokenize(text: str) -> List[str]:
    """Tokenize text into sentences using NLTK.

    Args:
        text: Input text to tokenize.

    Returns:
        List of original (uncleaned) sentences, filtered to remove
        sentences shorter than 10 characters.
    """
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        download_nltk_resources()
        sentences = sent_tokenize(text)

    # Filter sentences shorter than 10 chars
    sentences = [s for s in sentences if len(s) >= 10]

    return sentences


def preprocess_for_tfidf(text: str) -> str:
    """Preprocess text for TF-IDF fitting.

    Args:
        text: Raw input text.

    Returns:
        Cleaned version of text suitable for TF-IDF fitting.
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

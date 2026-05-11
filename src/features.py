"""
Feature extraction for extractive summarization.

This module provides TWO different ways to score sentences:

1. TF-IDF: Counts how important each word is in the document
   - Good at finding topic-specific keywords
   - Fast, simple

2. Sentence Embeddings: Measures semantic similarity
   - Good at finding sentences that capture the main idea
   - Understands synonyms and related concepts

WHY BOTH?
=========
TF-IDF finds important WORDS.
Embeddings find similar MEANINGS.

A sentence might have important words (high TF-IDF) but not capture
the main topic (low embedding similarity). Combining both gives
the best of both worlds.
"""

import pickle
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports (from src package)
from . import config
from . import preprocessing


class TFIDFExtractor:
    """
    TF-IDF based sentence scoring.

    WHAT IS TF-IDF?
    ===============
    TF-IDF = Term Frequency × Inverse Document Frequency

    - Term Frequency (TF): How often a word appears in THIS document
      (high = important to this document)

    - Inverse Document Frequency (IDF): How rare/common the word is ACROSS documents
      (high = rare word, more informative)

    - TF × IDF = High score for words that appear often in THIS doc but rarely in others

    Example:
    - "the" appears in EVERY document → low IDF → low score
    - "machine" appears in tech articles → high IDF → high score

    HOW IT WORKS HERE
    =================
    1. Fit: Learn vocabulary from training documents
    2. Transform: Convert sentences to TF-IDF vectors
    3. Score: Sum up TF-IDF values for each sentence

    A sentence with important keywords gets a high score.
    """

    def __init__(self):
        """
        Initialize TF-IDF vectorizer.

        max_features=10000: Only keep top 10,000 words
        This prevents memory issues with large vocabularies.
        """
        self.vectorizer = TfidfVectorizer(max_features=10000)

    def fit(self, texts: List[str]) -> None:
        """
        Learn vocabulary and IDF from training documents.

        This is CRITICAL: TF-IDF must be fit ONLY on training data!
        If you fit on test data, you leak information.

        Args:
            texts: List of documents to learn vocabulary from.
                  Should be CLEANED (lowercase, no punctuation).
        """
        # Clean each text before fitting
        cleaned_texts = [preprocessing.preprocess_for_tfidf(t) for t in texts]
        self.vectorizer.fit(cleaned_texts)

    def score_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Score each sentence by sum of TF-IDF values.

        A sentence's score = sum of TF-IDF for all words in it.
        Sentences with more important words get higher scores.

        Args:
            sentences: List of sentences to score.

        Returns:
            Array of scores (higher = more important words).
        """
        # Clean sentences for TF-IDF
        cleaned_sentences = [preprocessing.preprocess_for_tfidf(s) for s in sentences]

        # Transform to TF-IDF vectors
        # Each row = one sentence, each column = one word
        tfidf_matrix = self.vectorizer.transform(cleaned_sentences)

        # Score = sum of all TF-IDF values in the sentence
        # More important words = higher sum
        scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

        return scores

    def save(self, path: Path) -> None:
        """Save fitted vectorizer for later use."""
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: Path) -> None:
        """Load saved vectorizer."""
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)


class EmbeddingScorer:
    """
    Sentence embedding similarity scoring.

    WHAT ARE EMBEDDINGS?
    ===================
    Embeddings are dense vectors (384 dimensions for MiniLM) that
    represent the MEANING of text.

    Similar meanings → similar vectors (close in vector space)
    Different meanings → different vectors (far apart)

    Example:
    - "dog" and "puppy" → similar vectors
    - "dog" and "car" → different vectors

    HOW IT WORKS HERE
    =================
    1. Encode the FULL DOCUMENT → document embedding
    2. Encode each SENTENCE → sentence embeddings
    3. Measure cosine similarity between each sentence and the document

    Sentences with high similarity capture the main topic!
    """

    def __init__(self, model_name: str = None):
        """
        Load sentence transformer model.

        Default: all-MiniLM-L6-v2
        - 384 dimensions
        - Fast inference
        - Good quality

        Alternative: all-mpnet-base-v2 (768 dimensions, better quality)
        """
        model_name = model_name or config.EMBED_MODEL
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Convert text to embedding vectors.

        Args:
            texts: List of texts to encode.

        Returns:
            Array of shape (len(texts), embedding_dim)
        """
        return self.model.encode(texts)

    def score_sentences(self, sentences: List[str], document: str) -> np.ndarray:
        """
        Score sentences by similarity to full document.

        This answers: "How well does this sentence represent the document?"

        Algorithm:
        1. Encode entire document → 1 embedding (384 dims)
        2. Encode each sentence → N embeddings (384 dims each)
        3. Compute cosine similarity between each sentence and document

        Cosine similarity:
        - 1.0 = identical meaning
        - 0.0 = completely different
        - -1.0 = opposite meaning

        Sentences with high similarity capture the main topic!

        Args:
            sentences: List of sentences to score.
            document: Full document text.

        Returns:
            Array of similarity scores (0-1, higher = more representative).
        """
        # Step 1: Encode the full document
        doc_embedding = self.model.encode([document])

        # Step 2: Encode all sentences
        sent_embeddings = self.model.encode(sentences)

        # Step 3: Cosine similarity between each sentence and document
        # For each sentence: how similar is it to the whole document?
        similarities = cosine_similarity(sent_embeddings, doc_embedding).flatten()

        return similarities


if __name__ == "__main__":
    # Test TFIDFExtractor
    print("Testing TFIDFExtractor...")
    extractor = TFIDFExtractor()

    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing deals with text data.",
    ]
    extractor.fit(train_texts)

    test_sentences = [
        "A quick fox jumps over a dog.",
        "AI uses machine learning techniques.",
        "This sentence is unrelated to the topic.",
    ]
    scores = extractor.score_sentences(test_sentences)
    print(f"TF-IDF scores: {scores}")

    # Test EmbeddingScorer
    print("\nTesting EmbeddingScorer...")
    scorer = EmbeddingScorer()

    doc = "Machine learning is a subset of artificial intelligence. It enables computers to learn from data."
    sents = [
        "Machine learning is a subset of AI.",
        "The sky is blue today.",
        "ML enables learning from data.",
    ]
    embed_scores = scorer.score_sentences(sents, doc)
    print(f"Embedding scores: {embed_scores}")

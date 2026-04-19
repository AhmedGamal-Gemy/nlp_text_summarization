"""Feature extraction for summarization.

Provides TF-IDF and sentence embedding-based feature extraction
for scoring sentences in extractive summarization.
"""

import pickle
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config
import preprocessing


class TFIDFExtractor:
    """TF-IDF based sentence scoring extractor.

    Uses sklearn TfidfVectorizer to extract TF-IDF features
    and score sentences based on term frequency weights.
    """

    def __init__(self):
        """Initialize TF-IDF vectorizer with default settings."""
        self.vectorizer = TfidfVectorizer(max_features=10000)

    def fit(self, texts: List[str]) -> None:
        """Fit TF-IDF vectorizer on list of cleaned texts.

        Args:
            texts: List of text strings to fit on (should be pre-cleaned).
        """
        cleaned_texts = [preprocessing.preprocess_for_tfidf(t) for t in texts]
        self.vectorizer.fit(cleaned_texts)

    def score_sentences(self, sentences: List[str]) -> np.ndarray:
        """Score sentences using TF-IDF weights.

        Args:
            sentences: List of sentences to score.

        Returns:
            1D numpy array of TF-IDF scores per sentence.
        """
        # Clean each sentence
        cleaned_sentences = [preprocessing.preprocess_for_tfidf(s) for s in sentences]

        # Transform with fitted vectorizer
        tfidf_matrix = self.vectorizer.transform(cleaned_sentences)

        # Return sum of TF-IDF weights per sentence
        scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

        return scores

    def save(self, path: Path) -> None:
        """Save vectorizer to pickle file.

        Args:
            path: Path to save pickle file.
        """
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: Path) -> None:
        """Load vectorizer from pickle file.

        Args:
            path: Path to pickle file.
        """
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)


class EmbeddingScorer:
    """Sentence embedding-based document-sentence similarity scorer.

    Uses sentence-transformers to encode sentences and documents,
    then calculates cosine similarity between them.
    """

    def __init__(self, model_name: str = None):
        """Initialize sentence transformer model.

        Args:
            model_name: Name of sentence-transformers model.
                       Defaults to config.EMBED_MODEL.
        """
        model_name = model_name or config.EMBED_MODEL
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using sentence transformer.

        Args:
            texts: List of text strings to encode.

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim).
        """
        return self.model.encode(texts)

    def score_sentences(self, sentences: List[str], document: str) -> np.ndarray:
        """Score sentences based on similarity to full document.

        Args:
            sentences: List of sentences to score.
            document: Full document text.

        Returns:
            1D numpy array of cosine similarity scores.
        """
        # Encode document
        doc_embedding = self.model.encode([document])

        # Encode sentences
        sent_embeddings = self.model.encode(sentences)

        # Calculate cosine similarity between each sentence and document
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

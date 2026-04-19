"""Extractive summarizer using TF-IDF and sentence embeddings.

Provides ExtractiveSummarizer class that combines TF-IDF and embedding
scores to select top-k sentences from an article.
"""

from typing import List

import numpy as np

import config
import preprocessing
from features import TFIDFExtractor, EmbeddingScorer


def normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range.

    Args:
        scores: Array of scores to normalize.

    Returns:
        Normalized scores in [0, 1] range.
    """
    min_val = np.min(scores)
    max_val = np.max(scores)

    if max_val == min_val:
        return np.ones_like(scores)

    return (scores - min_val) / (max_val - min_val)


class ExtractiveSummarizer:
    """Extractive summarizer combining TF-IDF and embedding scores.

    Uses TF-IDF term frequency weights and sentence embedding
    similarity to score and rank sentences, then selects
    the top-k most important sentences.
    """

    def __init__(self, tfidf: TFIDFExtractor, embedder: EmbeddingScorer):
        """Initialize extractive summarizer.

        Args:
            tfidf: Fitted TFIDFExtractor instance.
            embedder: EmbeddingScorer instance.
        """
        self.tfidf = tfidf
        self.embedder = embedder

    def summarize(self, article: str, top_k: int = None) -> str:
        """Generate extractive summary by selecting top-k sentences.

        Args:
            article: Input article text.
            top_k: Number of sentences to include in summary.
                   Defaults to config.TOP_K_SENTENCES.

        Returns:
            Extracted sentences joined by space, ordered by
            their position in the original article.
        """
        top_k = top_k or config.TOP_K_SENTENCES

        # Step 1: Tokenize into sentences
        sentences = preprocessing.sentence_tokenize(article)

        if len(sentences) <= top_k:
            return " ".join(sentences)

        # Step 2: Score with TF-IDF
        tfidf_scores = self.tfidf.score_sentences(sentences)
        tfidf_scores = normalize(tfidf_scores)

        # Step 3: Score with embeddings
        embed_scores = self.embedder.score_sentences(sentences, article)
        embed_scores = normalize(embed_scores)

        # Step 4: Combine scores
        final_scores = (
            config.TFIDF_WEIGHT * tfidf_scores + config.EMBED_WEIGHT * embed_scores
        )

        # Step 5: Get top-k indices (sorted by score descending)
        top_indices = np.argsort(final_scores)[-top_k:]

        # Step 6: Sort indices to restore original order
        sorted_indices = sorted(top_indices)

        # Step 7: Join sentences in original order
        summary = " ".join(sentences[i] for i in sorted_indices)

        return summary


if __name__ == "__main__":
    # Test ExtractiveSummarizer
    print("Testing ExtractiveSummarizer...")

    # Create and fit TF-IDF
    tfidf = TFIDFExtractor()
    train_texts = [
        "The quick brown fox jumps over the lazy dog. It was a sunny day.",
        "Machine learning is a subset of artificial intelligence. Deep learning is a subfield.",
        "Natural language processing deals with text data. Text summarization is an NLP task.",
    ]
    tfidf.fit(train_texts)

    # Create embedder
    embedder = EmbeddingScorer()

    # Create summarizer
    summarizer = ExtractiveSummarizer(tfidf, embedder)

    # Test article
    article = (
        "Machine learning is a subset of artificial intelligence. "
        "It enables computers to learn from data without being explicitly programmed. "
        "Deep learning uses neural networks with multiple layers. "
        "The quick brown fox jumped over the lazy dog yesterday. "
        "Natural language processing is another important field."
    )

    summary = summarizer.summarize(article, top_k=3)
    print(f"Article: {article[:100]}...")
    print(f"Summary: {summary}")

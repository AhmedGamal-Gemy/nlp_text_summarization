"""
Extractive summarizer using TF-IDF and sentence embeddings.

WHAT IS EXTRACTIVE SUMMARIZATION?
=================================
Extractive = "extracting" sentences from the original text.

Instead of generating new text (like BART), we select the most
important sentences from the original article and combine them.

Pros:
- 100% faithful to original (no hallucination)
- Fast (no neural generation)
- Works on CPU
- Interpretable (can see which sentences were selected)

Cons:
- May be less fluent
- Can't paraphrase
- Limited by what's in the original text

HOW IT WORKS
============
1. Split article into sentences
2. Score each sentence by importance (TF-IDF + embeddings)
3. Pick top-k highest scoring sentences
4. Restore original order (important for readability!)
5. Join into summary
"""

from typing import List

import numpy as np

import config
import preprocessing
from features import TFIDFExtractor, EmbeddingScorer


def normalize(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] range.

    This ensures TF-IDF scores and embedding scores are on the same scale
    before combining them. Otherwise, one could dominate the other.

    Formula: (x - min) / (max - min)

    Example:
        Input:  [1, 2, 3, 4, 5]
        Output: [0, 0.25, 0.5, 0.75, 1.0]

    If all scores are equal, return all 1.0 (to avoid division by zero).
    """
    min_val = np.min(scores)
    max_val = np.max(scores)

    # Handle edge case: all scores equal
    if max_val == min_val:
        return np.ones_like(scores)

    # Normalize to [0, 1]
    return (scores - min_val) / (max_val - min_val)


class ExtractiveSummarizer:
    """
    Combines TF-IDF and embedding scores to select best sentences.

    The key insight: combine two complementary scoring methods:
    - TF-IDF finds important words
    - Embeddings find semantically central sentences

    Together they capture both keyword importance AND semantic relevance.
    """

    def __init__(self, tfidf: TFIDFExtractor, embedder: EmbeddingScorer):
        """
        Initialize with fitted feature extractors.

        Args:
            tfidf: Fitted TFIDFExtractor (call fit() first!)
            embedder: EmbeddingScorer (loads model)
        """
        self.tfidf = tfidf
        self.embedder = embedder

    def summarize(self, article: str, top_k: int = None) -> str:
        """
        Generate extractive summary.

        THE ALGORITHM:
        ============
        Step 1: Split article into sentences
        Step 2: Score each sentence with TF-IDF
        Step 3: Score each sentence with embeddings
        Step 4: Normalize both scores to [0,1]
        Step 5: Combine with weights (default: 50/50)
        Step 6: Pick top-k highest scoring sentences
        Step 7: Sort by original position (readability!)
        Step 8: Join into final summary

        CRITICAL: Step 7!
        If we just output top-k by score, sentences would be out of order.
        The summary would jump around randomly. Restoring original
        order makes it readable.

        Args:
            article: Input text to summarize.
            top_k: How many sentences to extract (default: 3).

        Returns:
            Selected sentences joined together, in original order.
        """
        top_k = top_k or config.TOP_K_SENTENCES

        # Step 1: Split into sentences
        sentences = preprocessing.sentence_tokenize(article)

        # Edge case: article has fewer sentences than requested
        if len(sentences) <= top_k:
            return " ".join(sentences)

        # Step 2: Score with TF-IDF
        # What does this tell us? Which sentences have important keywords
        tfidf_scores = self.tfidf.score_sentences(sentences)
        tfidf_scores = normalize(tfidf_scores)

        # Step 3: Score with embeddings
        # What does this tell us? Which sentences capture the main topic
        embed_scores = self.embedder.score_sentences(sentences, article)
        embed_scores = normalize(embed_scores)

        # Step 4: Combine scores
        # Weight by config (default: 50% TF-IDF + 50% embedding)
        # This balances keyword importance vs semantic relevance
        final_scores = (
            config.TFIDF_WEIGHT * tfidf_scores + config.EMBED_WEIGHT * embed_scores
        )

        # Step 5: Get indices of top-k sentences
        # np.argsort sorts ASCENDING, so we take last k
        top_indices = np.argsort(final_scores)[-top_k:]

        # Step 6: Restore original order
        # CRITICAL! Without this, output would be scrambled
        sorted_indices = sorted(top_indices)

        # Step 7: Build final summary
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

"""CLI demo for text summarization.

Interactive CLI loop accepting article text and
running both extractive and abstractive models.
"""

import sys
import time

import config
import preprocessing
from features import TFIDFExtractor, EmbeddingScorer
from extractive import ExtractiveSummarizer
from abstractive import AbstractiveSummarizer


def print_header():
    """Print demo header."""
    print("=" * 70)
    print("Text Summarization CLI Demo")
    print("=" * 70)
    print("\nEnter article text to summarize. Type 'quit' to exit.")
    print("-" * 70)


def run_demo():
    """Run interactive demo loop."""
    # Initialize models
    print("\nInitializing models...")

    # Preprocess
    preprocessing.download_nltk_resources()

    # Extractive
    print("  Loading Extractive model...")
    tfidf = TFIDFExtractor()
    embedder = EmbeddingScorer()
    ext_summarizer = ExtractiveSummarizer(tfidf, embedder)

    # Abstractive
    print("  Loading Abstractive model...")
    try:
        abs_summarizer = AbstractiveSummarizer()
    except Exception as e:
        print(f"  Warning: Could not load BART: {e}")
        print("  Abstractive model may not work")
        abs_summarizer = None

    # Fit TF-IDF on simple corpus
    print("  Fitting TF-IDF on training data...")
    sample_corpus = [
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing deals with text data.",
        "Deep learning uses neural networks with multiple layers.",
    ]
    tfidf.fit(sample_corpus)

    print("\nModels ready!")
    print("-" * 70)

    # Demo loop
    while True:
        try:
            # Get input
            print("\n")
            article = input("Enter article text: ").strip()

            # Check quit
            if article.lower() == "quit":
                print("Exiting...")
                break

            # Skip empty
            if not article:
                continue

            # Run extractive
            print("\n[1] Extractive Summary:")
            start = time.time()
            try:
                ext_summary = ext_summarizer.summarize(article)
                ext_time = time.time() - start
                print(f"  {ext_summary}")
                ext_words = len(ext_summary.split())
                orig_words = len(article.split())
                ext_ratio = ext_words / orig_words if orig_words > 0 else 0
                print(f"  Words: {ext_words} (compression: {ext_ratio:.2%})")
                print(f"  Time: {ext_time:.2f}s")
            except Exception as e:
                print(f"  Error: {e}")

            # Run abstractive
            if abs_summarizer:
                print("\n[2] Abstractive Summary:")
                start = time.time()
                try:
                    abs_summary = abs_summarizer.summarize(article)
                    abs_time = time.time() - start
                    print(f"  {abs_summary}")
                    abs_words = len(abs_summary.split())
                    abs_ratio = abs_words / orig_words if orig_words > 0 else 0
                    print(f"  Words: {abs_words} (compression: {abs_ratio:.2%})")
                    print(f"  Time: {abs_time:.2f}s")
                except Exception as e:
                    print(f"  Error: {e}")

            print("-" * 70)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            break

    print("\nDone!")


if __name__ == "__main__":
    print_header()
    run_demo()

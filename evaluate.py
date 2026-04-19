"""Evaluation pipeline for summarization models.

Provides ROUGE scoring, compression ratio calculation,
model comparison, and qualitative examples.
"""

from typing import List, Dict, Any
import time

import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer

import config
import preprocessing
from features import TFIDFExtractor, EmbeddingScorer
from extractive import ExtractiveSummarizer
from abstractive import AbstractiveSummarizer


def compute_rouge(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores for hypotheses vs references.

    Args:
        hypotheses: List of generated summaries.
        references: List of reference summaries.

    Returns:
        Dictionary with F-measure for rouge1, rouge2, rougeL.
    """
    scorer = rouge_scorer.RougeScorer(config.ROUGE_TYPES, use_stemmer=True)

    # Collect scores
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores),
    }


def compression_ratio(original: str, summary: str) -> float:
    """Calculate compression ratio (summary words / original words).

    Args:
        original: Original article text.
        summary: Generated summary text.

    Returns:
        Compression ratio as float.
    """
    original_words = len(original.split())
    summary_words = len(summary.split())

    if original_words == 0:
        return 0.0

    return summary_words / original_words


def evaluate_model(
    summarizer, dataset_samples: List[Dict[str, Any]], model_name: str
) -> Dict[str, Any]:
    """Evaluate summarizer on dataset samples.

    Args:
        summarizer: Summarizer instance (extractive or abstractive).
        dataset_samples: List of dataset samples with 'article' and 'highlights'.
        model_name: Name of the model for reporting.

    Returns:
        Dictionary with ROUGE scores and compression ratio.
    """
    print(f"\nEvaluating {model_name}...")

    hypotheses = []
    references = []
    compression_ratios = []

    # Test on first 200 samples
    test_samples = dataset_samples[:200]

    start_time = time.time()

    for i, sample in enumerate(test_samples):
        article = sample["article"]
        reference = sample["highlights"]

        try:
            if isinstance(summarizer, ExtractiveSummarizer):
                summary = summarizer.summarize(article)
            else:
                summary = summarizer.summarize(article)

            hypotheses.append(summary)
            references.append(reference)
            compression_ratios.append(compression_ratio(article, summary))

        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_samples)}")

    elapsed = time.time() - start_time

    # Compute ROUGE
    rouge_scores = compute_rouge(hypotheses, references)
    avg_compression = np.mean(compression_ratios)

    # Print results
    print(f"\n{model_name} Results:")
    print("=" * 50)
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"Compression Ratio: {avg_compression:.4f}")
    print(f"Time: {elapsed:.2f}s")

    return {
        "model_name": model_name,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "compression_ratio": avg_compression,
        "time": elapsed,
    }


def compare_models(
    extractive_results: Dict[str, Any], abstractive_results: Dict[str, Any]
) -> None:
    """Print side-by-side comparison of model results.

    Args:
        extractive_results: Results from evaluate_model for extractive.
        abstractive_results: Results from evaluate_model for abstractive.
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # Header
    print(f"{'Metric':<25} {'Extractive':>20} {'Abstractive':>20}")
    print("-" * 70)

    # ROUGE-1
    print(
        f"{'ROUGE-1':<25} {extractive_results['rouge1']:>20.4f} {abstractive_results['rouge1']:>20.4f}"
    )

    # ROUGE-2
    print(
        f"{'ROUGE-2':<25} {extractive_results['rouge2']:>20.4f} {abstractive_results['rouge2']:>20.4f}"
    )

    # ROUGE-L
    print(
        f"{'ROUGE-L':<25} {extractive_results['rougeL']:>20.4f} {abstractive_results['rougeL']:>20.4f}"
    )

    # Compression Ratio
    print(
        f"{'Compression Ratio':<25} {extractive_results['compression_ratio']:>20.4f} {abstractive_results['compression_ratio']:>20.4f}"
    )

    # Time
    print(
        f"{'Time (seconds)':<25} {extractive_results['time']:>20.2f} {abstractive_results['time']:>20.2f}"
    )

    print("=" * 70)


def qualitative_examples(
    ext_summarizer: ExtractiveSummarizer,
    abs_summarizer: AbstractiveSummarizer,
    samples: List[Dict[str, Any]],
    n: int = 5,
) -> None:
    """Print qualitative examples comparing model outputs.

    Args:
        ext_summarizer: ExtractiveSummarizer instance.
        abs_summarizer: AbstractiveSummarizer instance.
        samples: List of dataset samples.
        n: Number of examples to show.
    """
    print("\n" + "=" * 70)
    print("QUALITATIVE EXAMPLES")
    print("=" * 70)

    for i, sample in enumerate(samples[:n]):
        print(f"\n--- Example {i + 1} ---")
        print(f"Article (first 200 chars): {sample['article'][:200]}...")
        print(f"\nReference: {sample['highlights']}")

        try:
            ext_summary = ext_summarizer.summarize(sample["article"])
            print(f"\nExtractive: {ext_summary}")

            abs_summary = abs_summarizer.summarize(sample["article"])
            print(f"\nAbstractive: {abs_summary}")

        except Exception as e:
            print(f"Error: {e}")

        print("-" * 70)


def main():
    """Main evaluation function."""
    # Download NLTK resources
    print("Downloading NLTK resources...")
    preprocessing.download_nltk_resources()

    # Load dataset
    print(f"Loading CNN/DailyMail dataset ({config.DATA_SAMPLES} samples)...")
    try:
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:5000]")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Split into train/val/test (80/10/10)
    train_size = int(0.8 * config.DATA_SAMPLES)
    val_size = int(0.1 * config.DATA_SAMPLES)

    train_data = dataset[:train_size]
    val_data = dataset[train_size : train_size + val_size]
    test_data = dataset[train_size + val_size : config.DATA_SAMPLES]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Get training articles for TF-IDF fitting
    train_articles = [sample["article"] for sample in train_data]

    # Initialize extractive model
    print("\nInitializing Extractive model...")
    tfidf = TFIDFExtractor()
    tfidf.fit(train_articles)

    embedder = EmbeddingScorer()
    ext_summarizer = ExtractiveSummarizer(tfidf, embedder)

    # Initialize abstractive model
    print("Initializing Abstractive model...")
    try:
        abs_summarizer = AbstractiveSummarizer()
    except Exception as e:
        print(f"Error loading BART: {e}")
        print("Skipping abstractive evaluation")
        abs_summarizer = None

    # Run evaluation
    print("\nRunning evaluation on test set...")

    test_articles = test_data

    # Evaluate extractive
    ext_results = evaluate_model(ext_summarizer, test_articles, "Extractive")

    # Evaluate abstractive
    if abs_summarizer:
        abs_results = evaluate_model(abs_summarizer, test_articles, "Abstractive")

        # Compare models
        compare_models(ext_results, abs_results)

        # Qualitative examples
        qualitative_examples(ext_summarizer, abs_summarizer, test_articles, n=5)
    else:
        print("Abstractive model not available")


if __name__ == "__main__":
    main()

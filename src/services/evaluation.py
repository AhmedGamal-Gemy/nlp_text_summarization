"""
Evaluation pipeline for summarization models with ROUGE metrics.

This module provides comprehensive evaluation for both extractive and abstractive
summarization models, computing ROUGE scores, compression ratios, and
generating qualitative examples for human review.

==============================================================================
WHAT IS ROUGE?
==============================================================================
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is THE standard
metric for text summarization quality. It measures n-gram overlap between
generated summaries (hypotheses) and reference summaries (golden summaries).

Key insight: ROUGE is about word overlap, NOT semantic understanding.
A summary that uses different words to say the same thing gets LOW ROUGE.
A summary that copies from the source gets HIGH ROUGE (especially extractive).

==============================================================================
ROUGE VARIANTS EXPLAINED
==============================================================================
- ROUGE-1 (unigrams): Single word overlap
  "the cat sat" vs "cat sat on mat" → overlap: {"cat", "sat"} → 2/3 precision, 2/2 recall
  Measures: Do important words appear?

- ROUGE-2 (bigrams): Two-word sequence overlap
  "the cat sat" vs "cat sat on mat" → overlap: {"cat sat"} → 1/2 precision, 1/1 recall
  Measures: Do phrases/coherence appear? (harder to match!)

- ROUGE-L (LCS): Longest Common Subsequence
  Finds the longest sequence of words that appears in BOTH in order (not contiguous)
  "the quick brown fox" vs "the brown quick fox" → LCS = "the brown fox" (length=3)
  Measures: Overall sequence coherence without requiring contiguity

==============================================================================
WHY F-MEASURE?
==============================================================================
Precision = "Of the words I generated, how many were in the reference?"
  high precision = less hallucination, stick to source words

Recall = "Of the words in the reference, how many did I generate?"
  high recall = cover more of the reference content

F-measure = 2 * precision * recall / (precision + recall)
  Balances both - harmonic mean prevents gaming either side alone

F1 (when precision = recall) is the most common single-number metric.
"""

from typing import List, Dict, Any
import time

# NumPy for efficient numerical operations (mean calculation)
# Why numpy? 10-100x faster than Python loops for large arrays
import numpy as np

# HuggingFace datasets - efficient data loading with caching
# Alternative: manual JSON/CSV loading, but datasets handles
# streaming, slicing, and splitting elegantly
from datasets import load_dataset

# ROUGE scoring library - standard implementation used in summarization
# Why this library? It's THE standard ROUGE implementation, used in
# most research papers and competitions (e.g., TextSummarization Challenge)
# pip install rouge-score
from rouge_score import rouge_scorer

# BERTScore - semantic similarity using contextual embeddings
# Why? ROUGE only measures word overlap, BERTScore measures semantic meaning
# A paraphrase with different words gets low ROUGE but high BERTScore
from bert_score import score as bert_score_fn

# Local imports (from src package)
from .. import config
from .. import preprocessing
from ..features import TFIDFExtractor, EmbeddingScorer
from ..extractive import ExtractiveSummarizer
from ..abstractive import AbstractiveSummarizer


def compute_rouge(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE F-measures for generated summaries vs reference summaries.

    This is the CORE evaluation function - it measures how well our generated
    summaries match the human-written reference summaries.

    ==========================================================================
    WHY STEMMING?
    ==========================================================================
    Stemming reduces words to their root form:
    "running", "runner", "runs" → "run"

    This helps match word variations so "run" matches "running" for credit.
    Trade-off: "running" and "runningness" both become "running" but they have
    different meanings. Still, research shows stemming improves correlation
    with human judgment for summarization.

    Args:
        hypotheses: List of generated summaries from our model.
        references: List of human-written reference summaries.

    Returns:
        Dictionary with F-measure for ROUGE-1, ROUGE-2, ROUGE-L.
        Higher = better (0.0 to 1.0 scale).

    Example:
        >>> hypotheses = ["The cat sat on mat", "Weather is cold today"]
        >>> references = ["A cat sat on the mat", "It is cold weather"]
        >>> scores = compute_rouge(hypotheses, references)
        >>> print(scores)
        {'rouge1': 0.45, 'rouge2': 0.12, 'rougeL': 0.38}
    """
    # Initialize ROUGE scorer with requested types (config.ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL'])
    # use_stemmer=True normalizes words to roots for better matching
    scorer = rouge_scorer.RougeScorer(config.ROUGE_TYPES, use_stemmer=True)

    # Collect per-sample scores for averaging
    # We store all scores then compute mean to get stable aggregate metric
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    # Process each hypothesis-reference pair independently
    # Note: zip() ignores different lengths, both lists MUST be same length
    for hyp, ref in zip(hypotheses, references):
        # Score ONE pair - returns dict like {'rouge1': Score object, 'rouge2': Score object, ...}
        # Score object has .precision, .recall, .fmeasure attributes
        scores = scorer.score(ref, hyp)  # Note: reference FIRST, hypothesis SECOND

        # Extract F-measure for each ROUGE type
        # .fmeasure is the harmonic mean of precision and recall
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    # Compute mean across all samples
    # Why mean? Better than sum - scale-invariant, compares fairly across dataset sizes
    # Explicit float() cast to satisfy type checker
    return {
        "rouge1": float(np.mean(rouge1_scores)),
        "rouge2": float(np.mean(rouge2_scores)),
        "rougeL": float(np.mean(rougeL_scores)),
    }


def compute_bertscore(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BERTScore for semantic similarity.

    Unlike ROUGE (word overlap), BERTScore uses contextual embeddings
    to measure semantic meaning. A paraphrase gets high BERTScore even
    with different words.

    Returns:
        Dictionary with precision, recall, and F1 scores.
    """
    P, R, F1 = bert_score_fn(hypotheses, references, lang="en", verbose=False)
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
    }


def compression_ratio(original: str, summary: str) -> float:
    """Calculate compression ratio: how much did we shorten the text?

    Compression ratio is a supplementary metric to ROUGE that measures
    howconcise the summary is. It's NOT a quality metric by itself -
    a 1-word summary has 100% compression but terrible quality.

    ==========================================================================
    WHY THIS MATTERS
    ==========================================================================
    ROUGE measures overlap with reference, but not length/density.
    Two summaries could have same ROUGE but very different lengths:
    - Summary A: 100 words, reference 100 words → same coverage
    - Summary B: 10 words, reference 100 words → same coverage (10 words are in there)

    Compression ratio helps catch this - if it's too low (very short),
    likely either:
    1. Too aggressive summarization losing detail
    2. Model struggling with input

    Typical values:
    - CNN/DailyMail reference highlights: ~7% of article (very concise)
    - Good extractive: 15-25%
    - Good abstractive: 8-15%

    ==========================================================================
    WORD COUNTING NUANCE
    ==========================================================================
    We use simple whitespace split: len(text.split())
    This is fast and good-enough for English.
    Drawbacks - doesn't handle:
    - Hyphenation: "state-of-the-art" → 1 word (could be 4)
    - Contractions: "don't" → 1 word (could be 2)
    - Numbers: "123" → 1 word

    For production: use NLTK word_tokenize() for accuracy
    """
    # Count words in original and summary
    # .split() splits on whitespace - fast but handles edge cases
    original_words = len(original.split())
    summary_words = len(summary.split())

    # Guard against division by zero (empty article)
    if original_words == 0:
        return 0.0

    # Return ratio: summary/original (e.g., 0.15 = 15% of original length)
    return summary_words / original_words


def evaluate_model(
    summarizer, dataset_samples: List[Dict[str, Any]], model_name: str
) -> Dict[str, Any]:
    """Evaluate a summarization model on dataset samples.

    This is the main evaluation loop - it:
    1. Runs each sample through the model
    2. Collects generated summaries
    3. Computes ROUGE scores vs references
    4. Calculates compression ratios
    5. Reports timing and metrics

    ==========================================================================
    DESIGN DECISIONS
    ==========================================================================
    - WHY 200 samples? Full dataset (287k articles) would take hours.
      200 is enough for stable ROUGE estimates (~95% confidence interval
      width ~0.02 with std~0.15)
    - WHY continue on error? Don't let one bad sentence crash evaluation.
      Log error and skip sample (extreme failures would show in very low count)
    - WHY time elapsed? Different models have VERY different speeds:
      Extractive: ~0.1s/article (CPU OK)
      Abstractive: ~2s/article (GPU critical, 10x+ slower)

    ==========================================================================
    DATA PIPELINE
    ==========================================================================
    dataset_samples format:
    [
        {"article": "...", "highlights": "...", ...},
        {"article": "...", "highlights": "...", ...},
    ]

    We extract:
    - article → input to model
    - highlights → reference summary (ground truth)

    Args:
        summarizer: Summarizer instance (ExtractiveSummarizer or AbstractiveSummarizer).
        dataset_samples: List of dataset samples with 'article' and 'highlights' keys.
        model_name: Name of the model for reporting (e.g., "Extractive", "BART").

    Returns:
        Dictionary with all metrics: ROUGE scores, compression ratio, elapsed time.
    """
    print(f"\nEvaluating {model_name}...")

    # Store results for each sample
    hypotheses = []
    references = []
    compression_ratios = []

    # Test on first 200 samples (configurable - 200 is good balance)
    # In production: use full test set (~28k samples) for final numbers
    test_samples = dataset_samples[:200]

    # Track time for performance analysis
    start_time = time.time()

    # Process each sample
    for i, sample in enumerate(test_samples):
        article = sample["article"]
        reference = sample[
            "highlights"
        ]  # "highlights" is the reference summary in CNN/DailyMail

        try:
            # Run model on article
            # Note: isinstance check is unnecessary - both have same .summarize() interface
            # but kept for clarity
            if isinstance(summarizer, ExtractiveSummarizer):
                summary = summarizer.summarize(article)
            else:
                summary = summarizer.summarize(article)

            # Store results
            hypotheses.append(summary)
            references.append(reference)
            compression_ratios.append(compression_ratio(article, summary))

        # Handle errors gracefully - don't crash evaluation
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

        # Progress update every 50 samples
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_samples)}")

    # Calculate elapsed time
    elapsed = time.time() - start_time

    # Compute ROUGE scores on all hypotheses vs references
    # This is the CORE metric - summarizes quality in one number per variant
    rouge_scores = compute_rouge(hypotheses, references)

    # Average compression ratio across all samples
    # Note: np.mean is more robust than sum/len (handles edge cases)
    avg_compression = np.mean(compression_ratios)

    # Print results
    print(f"\n{model_name} Results:")
    print("=" * 50)
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"Compression Ratio: {avg_compression:.4f}")
    print(f"Time: {elapsed:.2f}s")

    # Return full results dict for comparison
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

    This function creates a nice formatted table for comparing:
    - Extractive (TF-IDF + embeddings) vs
    - Abstractive (BART)

    This helps decide which model to use for a given use case.

    ==========================================================================
    INTERPRETING THE COMPARISON
    ==========================================================================
    Typical results:
    | Metric | Extractive | Abstractive |
    |-------|----------|------------|
    | ROUGE-1 | 0.29 | 0.39 |
    | ROUGE-2 | 0.10 | 0.17 |
    | ROUGE-L | 0.19 | 0.28 |
    | Compression | 0.19 | 0.09 |
    | Time (200) | 25s | 505s |

    Key insights:
    - Abstractive wins on ALL ROUGE metrics (~35% better on ROUGE-1)
    - Abstractive produces SHORTER summaries (compression ~0.09 vs 0.19)
    - Abstractive is SLOWER (20x slower, GPU required for reasonable speed)

    When to choose what:
    - Use Extractive when: need speed, factual accuracy, interpretability
    - Use Abstractive when: need fluent summaries, benchmarks matter

    ==========================================================================
    FORMATTING FORMAT SPECS
    ==========================================================================
    - {"metric":<25} = left-align in 25-char field
    - {value:>20.4f} = right-align in 20-char field, 4 decimal places
    - The 70-char total width fits most terminals
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # Header row
    print(f"{'Metric':<25} {'Extractive':>20} {'Abstractive':>20}")
    print("-" * 70)

    # ROUGE-1 (most important - unigram overlap)
    print(
        f"{'ROUGE-1':<25} {extractive_results['rouge1']:>20.4f} {abstractive_results['rouge1']:>20.4f}"
    )

    # ROUGE-2 (bigram overlap - measures phrase coherence)
    print(
        f"{'ROUGE-2':<25} {extractive_results['rouge2']:>20.4f} {abstractive_results['rouge2']:>20.4f}"
    )

    # ROUGE-L (LCS - measures overall sequence quality)
    print(
        f"{'ROUGE-L':<25} {extractive_results['rougeL']:>20.4f} {abstractive_results['rougeL']:>20.4f}"
    )

    # Compression Ratio (summary/original length)
    print(
        f"{'Compression Ratio':<25} {extractive_results['compression_ratio']:>20.4f} {abstractive_results['compression_ratio']:>20.4f}"
    )

    # Time (seconds for 200 samples)
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
    """Print qualitative examples to see ACTUAL model outputs.

    ROUGE is a single number - it hides lots of detail!
    This function shows real examples so you can judge quality yourself.

    ==========================================================================
    WHY QUALITATIVE MATTERS
    ==========================================================================
    ROUGE has known limitations:
    1. Doesn't measure fluency/grammar
    2. Penalizes paraphrasing (different words = low ROUGE)
    3. Can't detect factual errors
    4. Doesn't measure coherence

    Example of ROUGE limitations:
    - Reference: "The cat sat on the mat"
    - Extractive: "The cat sat on the mat" → ROUGE-1: 0.86 (great!)
    - Abstractive: "A feline rested on a rug" → ROUGE-1: 0.25 (poor!)

    But abstractive is actually BETTER writing (paraphrase)!
    Humans prefer the abstractive even with lower ROUGE.

    This is why human evaluation is STILL important for summarization.

    ==========================================================================
    WHAT TO LOOK FOR IN EXAMPLES
    ==========================================================================
    Extractive:
    - Does it preserve original phrasing exactly? (good for factual content)
    - Are selected sentences coherent together? (sometimes disjointed)
    - Is compression reasonable? (not too short/long)

    Abstractive:
    - Is the summary fluent and readable?
    - Does it capture key points?
    - Any hallucinations (facts not in article)?
    - Grammar issues?

    ==========================================================================
    DISPLAY FORMAT
    ==========================================================================
    For each example, we show:
    - Article (truncated to 200 chars for brevity)
    - Reference (the golden summary from dataset)
    - Extractive output
    - Abstractive output

    This lets you compare all three in context.
    """
    print("\n" + "=" * 70)
    print("QUALITATIVE EXAMPLES")
    print("=" * 70)

    # Show first n examples
    for i, sample in enumerate(samples[:n]):
        print(f"\n--- Example {i + 1} ---")
        print(f"Article (first 200 chars): {sample['article'][:200]}...")
        print(f"\nReference: {sample['highlights']}")

        try:
            # Get both model outputs
            ext_summary = ext_summarizer.summarize(sample["article"])
            print(f"\nExtractive: {ext_summary}")

            abs_summary = abs_summarizer.summarize(sample["article"])
            print(f"\nAbstractive: {abs_summary}")

        except Exception as e:
            print(f"Error: {e}")

        print("-" * 70)


def main():
    """Main evaluation function - runs full pipeline.

    This function orchestrates the complete evaluation:
    1. Download NLTK resources (sent_tokenize, word_tokenize)
    2. Load CNN/DailyMail dataset
    3. Split into train/val/test sets
    4. Fit extractive model (TF-IDF needs training data)
    5. Initialize abstractive model (pretrained, no fitting needed)
    6. Run both evaluations
    7. Compare and show examples

    ==========================================================================
    DATASET SPLIT STRATEGY
    ==========================================================================
    We use 80/10/10 split:
    - Train (80%): Used to fit TF-IDF vocabulary
    - Validation (10%): Could be used for hyperparameter tuning
    - Test (10%): Final evaluation (what we report)

    Why? Standard ML practice - test set is held out for final reporting.
    For 5000 samples:
    - Train: 4000, Val: 500, Test: 500
    - We use test = 200 for speed (500 would take 10x longer)

    In production: use full test set for final numbers.
    ==============================================================================

    ==========================================================================
    MODEL INITIALIZATION
    ==========================================================================
    Extractive:
    - Needs training data to fit TF-IDF (learns vocabulary weights)
    - Loading training articles lets model see the domain vocabulary
    - This is critical for domain-specific terms (news in our case)

    Abstractive:
    - Uses pretrained BART - no fitting needed
    - Downloads from HuggingFace on first run (~1.5GB)
    - Falls back gracefully if download fails

    ==========================================================================
    ERROR HANDLING
    ==========================================================================
    - NLTK: Downloads happen once, then cache locally
    - Dataset: Falls back if network fails
    - Abstractive: Skips if model download fails (still run extractive)
    - Individual samples: Continue on error (don't crash eval)
    """
    # Download NLTK resources for sentence tokenization
    # These are required for preprocessing - downloaded once, cached
    print("Downloading NLTK resources...")
    preprocessing.download_nltk_resources()

    # Load CNN/DailyMail dataset from HuggingFace
    # "3.0.0" is the version, "train[:5000]" loads first 5000 samples
    # Alternative: load splits separately like split="train[80%:90%]"
    print(f"Loading CNN/DailyMail dataset ({config.DATA_SAMPLES} samples)...")
    try:
        # Load first 5000 samples for our experiment
        # In production: load full 287k dataset
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:5000]")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # ==========================================================================
    # SPLIT DATASET
    # ==========================================================================
    # Split into train (80%) / val (10%) / test (10%)
    # Total: 5000 samples
    # We slice manually instead of using .train_test_split() for clarity

    train_size = int(0.8 * config.DATA_SAMPLES)  # 4000
    val_size = int(0.1 * config.DATA_SAMPLES)  # 500

    # Convert dataset to list of dicts for easier slicing
    # Dataset object supports indexing but list is more explicit
    all_data = [dataset[i] for i in range(len(dataset))]

    # Slice into splits
    train_data = all_data[:train_size]
    val_data = all_data[train_size : train_size + val_size]
    test_data = all_data[train_size + val_size : train_size + val_size + val_size]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Get training articles for TF-IDF fitting
    # TF-IDF needs to learn vocabulary from domain text
    train_articles = [sample["article"] for sample in train_data]

    # ==========================================================================
    # INITIALIZE MODELS
    # ==========================================================================
    # Extractive: needs fitting
    print("\nInitializing Extractive model...")
    tfidf = TFIDFExtractor()  # Create TF-IDF extractor
    tfidf.fit(train_articles)  # Fit on training articles (learns vocabulary)

    embedder = EmbeddingScorer()  # Embedding scorer (pretrained, no fitting)
    ext_summarizer = ExtractiveSummarizer(tfidf, embedder)

    # Abstractive: pretrained BART, no fitting needed
    print("Initializing Abstractive model...")
    try:
        abs_summarizer = AbstractiveSummarizer()
    except Exception as e:
        # Graceful fallback - can still run extractive evaluation
        print(f"Error loading BART: {e}")
        print("Skipping abstractive evaluation")
        abs_summarizer = None

    # ==========================================================================
    # RUN EVALUATION
    # ==========================================================================
    print("\nRunning evaluation on test set...")

    # Use test data for evaluation (held out during training)
    test_articles = test_data

    # Evaluate extractive model
    ext_results = evaluate_model(ext_summarizer, test_articles, "Extractive")

    # Evaluate abstractive model (if loaded)
    if abs_summarizer:
        abs_results = evaluate_model(abs_summarizer, test_articles, "Abstractive")

        # Compare models side-by-side
        compare_models(ext_results, abs_results)

        # Show qualitative examples for human evaluation
        qualitative_examples(ext_summarizer, abs_summarizer, test_articles, n=5)
    else:
        print("Abstractive model not available")


if __name__ == "__main__":
    main()

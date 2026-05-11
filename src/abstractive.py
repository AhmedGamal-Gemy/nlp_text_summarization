"""
Abstractive summarizer using BART.

WHAT IS ABSTRACTIVE SUMMARIZATION?
=================================
Abstractive = "generating" new text, not just extracting.

BART reads the article and GENERATES a new summary from scratch.
The summary is not composed of original sentences - it's new text
that captures the key information.

HOW IT WORKS
============
BART (Bidirectional and Auto-Regressive Transformers):
1. Encoder: Reads the article (bidirectional attention)
2. Decoder: Generates the summary (autoregressive)

KEY GENERATION PARAMETERS:
- max_length: Don't generate more than X tokens
- min_length: Generate at least X tokens
- num_beams: Try multiple hypotheses, pick best
- length_penalty: Longer summaries preferred (>1.0) or penalized (<1.0)
- no_repeat_ngram_size: Don't repeat n-grams of this size (prevents repetition)

CRITICAL: Pass RAW text to BART, not cleaned!
BART was trained on natural text with punctuation.
Cleaning hurts performance.
"""

from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Local imports (from src package)
from . import config


class AbstractiveSummarizer:
    """
    BART-based abstractive summarizer.

    BART is a seq2seq (sequence-to-sequence) model:
    - Input: article text
    - Output: generated summary

    We use facebook/bart-large-cnn because:
    - "cnn" = fine-tuned on CNN/DailyMail dataset
    - This is THE dataset we're evaluating on!
    - It already knows the style of news summaries
    """

    def __init__(self, model_path: str = None):
        """
        Load BART model and tokenizer.

        Can load either:
        - Pretrained facebook/bart-large-cnn (default)
        - Fine-tuned checkpoint (if you've run train.py)

        Args:
            model_path: Optional path to fine-tuned model.
        """
        model_name = model_path if model_path else config.BART_MODEL

        # Tokenizer: converts text ↔ token IDs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Model: generates text
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Move to GPU if available
        self.model.to(config.DEVICE)

        # Evaluation mode (faster, no dropout)
        self.model.eval()

        # Track if using fine-tuned model
        self.is_finetuned = model_path is not None

    def summarize(self, article: str) -> str:
        """
        Generate summary for one article.

        THE GENERATION PROCESS:
        ======================
        1. Tokenize: Convert text to token IDs
        2. Encode: Run through encoder
        3. Generate: Autoregressive decoding with beam search
        4. Decode: Convert token IDs back to text

        WHAT IS BEAM SEARCH?
        ===================
        Instead of greedily picking the most likely next token,
        beam search keeps multiple hypotheses (beams) and
        picks the one with highest total probability.

        num_beams=4 means: keep 4 best partial summaries,
        pick the best final one.

        Args:
            article: RAW text (not cleaned!)

        Returns:
            Generated summary text.
        """
        # Step 1: Tokenize
        # Convert text → token IDs
        inputs = self.tokenizer(
            article,
            max_length=config.MAX_ARTICLE_LEN,  # Truncate if too long
            truncation=True,
            return_tensors="pt",  # PyTorch tensors
        )

        # Step 2: Move to GPU
        # Transfer input tensors to GPU/CPU
        inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

        # Step 3: Generate
        with torch.no_grad():  # No gradient needed for inference
            summary_ids = self.model.generate(
                **inputs,
                max_length=config.BART_MAX_LEN,  # Don't exceed 130 tokens
                min_length=config.BART_MIN_LEN,  # At least 30 tokens
                num_beams=config.BART_BEAMS,  # Beam search with 4 beams
                length_penalty=2.0,  # Prefer longer summaries
                no_repeat_ngram_size=3,  # No repeated 3-grams
                early_stopping=True,  # Stop when done
            )

        # Step 4: Decode
        # Convert token IDs → text
        # skip_special_tokens removes <s>, </s>, etc.
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

    def summarize_batch(self, articles: List[str], batch_size: int = 8) -> List[str]:
        """
        Generate summaries for multiple articles.

        Processes in batches for efficiency.

        Args:
            articles: List of article texts.
            batch_size: How many to process at once.

        Returns:
            List of generated summaries.
        """
        summaries = []

        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                padding=True,  # Pad to same length
                truncation=True,
                max_length=config.MAX_ARTICLE_LEN,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                summary_ids = self.model.generate(
                    **inputs,
                    max_length=config.BART_MAX_LEN,
                    min_length=config.BART_MIN_LEN,
                    num_beams=config.BART_BEAMS,
                    length_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            # Decode batch
            batch_summaries = self.tokenizer.batch_decode(
                summary_ids, skip_special_tokens=True
            )

            summaries.extend(batch_summaries)

        return summaries


if __name__ == "__main__":
    # Test AbstractiveSummarizer
    print("Testing AbstractiveSummarizer...")

    try:
        summarizer = AbstractiveSummarizer()

        # Test article
        article = (
            "Machine learning is a subset of artificial intelligence that "
            "enables computers to learn from data without being explicitly programmed. "
            "Deep learning uses neural networks with multiple layers to learn "
            "representations of data. Natural language processing handles "
            "text and speech data for various applications."
        )

        summary = summarizer.summarize(article)
        print(f"Article: {article}")
        print(f"Summary: {summary}")

    except Exception as e:
        print(f"Error (expected if models not installed): {e}")

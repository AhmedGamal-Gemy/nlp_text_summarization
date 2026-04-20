"""Abstractive summarizer using BART.

Provides AbstractiveSummarizer class that uses facebook/bart-large-cnn
pretrained model for abstractive summarization.
"""

from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import config


class AbstractiveSummarizer:
    """Abstractive summarizer using BART sequence-to-sequence model.

    Uses facebook/bart-large-cnn pretrained on CNN/DailyMail for
    generating fluent abstractive summaries.
    Can also load fine-tuned checkpoints.
    """

    def __init__(self, model_path: str = None):
        """Initialize BART model and tokenizer.

        Args:
            model_path: Optional path to fine-tuned model checkpoint.
                        If None, uses pretrained facebook/bart-large-cnn.
        """
        model_name = model_path if model_path else config.BART_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(config.DEVICE)
        self.model.eval()
        self.is_finetuned = model_path is not None
        if self.is_finetuned:
            print(f"Loaded fine-tuned model from: {model_path}")
        else:
            print(f"Loaded pretrained model: {config.BART_MODEL}")

    def summarize(self, article: str) -> str:
        """Generate abstractive summary for a single article.

        Args:
            article: Input article text (raw, not preprocessed).

        Returns:
            Generated fluent summary text.
        """
        # Tokenize and prepare inputs
        inputs = self.tokenizer(
            article,
            max_length=config.MAX_ARTICLE_LEN,
            truncation=True,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

        # Generate summary with beam search
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

        # Decode and return
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

    def summarize_batch(self, articles: List[str], batch_size: int = 8) -> List[str]:
        """Generate abstractive summaries for multiple articles.

        Args:
            articles: List of input article texts.
            batch_size: Number of articles to process at once.

        Returns:
            List of generated summary texts.
        """
        summaries = []

        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.MAX_ARTICLE_LEN,
                return_tensors="pt",
            )

            # Move inputs to device
            inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

            # Generate summaries
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

            # Decode
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

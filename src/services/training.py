"""
Fine-tuning script for BART summarization.

This script fine-tunes facebook/bart-large-cnn on CNN/DailyMail to improve
summarization quality. Fine-tuning adapts a pretrained model to a specific task.

==============================================================================
WHAT IS FINE-TUNING?
==============================================================================
Fine-tuning is TRANSFER LEARNING - taking a model pretrained on A and adapting
it for task B, using lessons from A to help with B.

Pretrained models (like BART) learn:
- General language understanding from massive text (books, web)
- Grammar, facts, reasoning, common patterns

Fine-tuning adds:
- Domain-specific knowledge (news articles → CNN/DailyMail style)
- Task-specific behavior (summarization)

==============================================================================
WHY FINE-TUNE VS PROMPT ENGINEERING?
==============================================================================
Prompt engineering (GPT-4 style):
- No training needed - just clever prompting
- Limited by model's knowledge cutoff
- Can't learn NEW patterns from your data
- Good for general knowledge, not domain-specific

Fine-tuning:
- Learns from YOUR data
- Adapts to YOUR format/style
- Better for specialized domains (legal, medical, news)
- Costs GPU time, but more accurate

==============================================================================
KEY FINE-TUNING CONCEPTS
==============================================================================
1. LEARNING RATE: Typically 10x-100x lower than training from scratch
   - Pretrained weights are already good
   - High LR would "unlearn" useful patterns
   - Our default: 3e-5 (0.00003)

2. WARMUP: Gradually increase LR at start
   - Why? Model parameters are random at init
   - High LR at start = chaotic gradients
   - Warmup gives time to stabilize
   - Our default: 10% of training steps

3. FEWER EPOCHS: Usually 2-5 instead of 10-100
   - Pretrained model already knows language
   - Just needs adjustment, not full training
   - More epochs = overfitting (memorizing training data)
   - Our default: 3 epochs

4. WEIGHT DECAY: Regularization to prevent overfitting
   - Penalizes large weights
   - Keeps model generalizable
   - Our default: 0.01 (1%)

==============================================================================
WHY SEQUENCE-TO-SEQUENCE?
==============================================================================
Summarization is a classic seq2seq task:
- Input: Article (~700 words)
- Output: Summary (~50 words)
- Both are sequences of tokens

BART (Bidirectional and Auto-Regressive Transformers) is seq2seq:
- Encoder: Reads input article (bidirectional attention)
- Decoder: Generates summary one token at a time (autoregressive)

Key insight: Autoregressive = generate one word at a time,
using previously generated words as context for next.

==============================================================================
WHY BART-LARGE-CNN SPECIFICALLY?
==============================================================================
We use facebook/bart-large-cnn not generic bart-large:
- Pretrained on CNN/DailyMail summarization
- Already knows news article format
- Better out-of-box for summarization
- 400M parameters (large but trainable on single GPU)

Trade-off: Other options:
- google/t5-large: More general, but needs more tuning
- google/pegasus-xsum: Better on extreme summarization
- Custom: Best if you have domain data (legal, medical)
"""

import argparse
from pathlib import Path

# PyTorch - deep learning framework
# Why PyTorch? Best ecosystem for transformer models,
# HuggingFace transformers built on PyTorch
import torch

# HuggingFace Transformers - pretrained models
# pip install transformers
# AutoTokenizer: Loads tokenizer from pretrained model
# AutoModelForSeq2SeqLM: Loads sequence-to-sequence model
# Seq2SeqTrainingArguments: Training configuration
# Seq2SeqTrainer: Training loop with seq2seq-specific features
# DataCollatorForSeq2Seq: Batches for seq2seq (handles padding)
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# Dataset loading - same as evaluation
from datasets import load_dataset

# Local imports (from src package)
from .. import config


def preprocess_function(
    examples, tokenizer, max_source_length=1024, max_target_length=130
):
    """Preprocess examples for BART fine-tuning.

    This function converts raw text (article, highlights) into token IDs
    that the model can process.

    ==========================================================================
    PREPROCESSING STEPS
    ==========================================================================
    1. Add task prefix: "summarize: " tells BART what to do
       - BART was trained with this prefix
       - Without it, quality degrades
    2. Tokenize input: Convert text → token IDs
       - max_length: Truncate to prevent OOM
       - padding: Pad to max_length for batching
    3. Tokenize labels: Same for target summary
       - These are what model tries to predict
    4. Return: Dict with input_ids, attention_mask, labels

    ==========================================================================
    WHY SPECIAL PREFIX?
    ==========================================================================
    BART was trained with "summarize: " prefix for summarization.
    This is a form of TASK PROMPTING - telling the model what task to do.

    Other tasks use different prefixes:
    - "summarize: " → summarization
    - "translate English to French: " → translation
    - "classify: " → classification

    ==========================================================================
    TOKENIZATION BASICS
    ==========================================================================
    Text: "The cat sat"
    ↓ Tokenizer (BART uses BPE)
    Token IDs: [72, 3987, 1012]
    ↓ Model
    Output: [72, 3987, 1012, 5, 890, ...] (next tokens)

    Vocabulary size: 50k tokens (BART)
    - Covers ~all English with subword units
    - Handles OOV words via subword decomposition
    - "unseen" → "un" + "seen"
    """
    # Add task prefix for summarization
    # This tells BART the task type - critical for quality
    inputs = ["summarize: " + doc for doc in examples["article"]]

    # Tokenize input articles
    # max_length=1024: BART's max input length
    # truncation=True: Cut off long articles (lossy but necessary)
    # padding="max_length": Pad to fixed length for batching
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        truncation=True,
        padding="max_length",
    )

    # Tokenize target summaries (highlights)
    # Same process as input, but for output
    # max_length=130: Typical CNN/DailyMail summary length
    labels = tokenizer(
        examples["highlights"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )

    # Store labels for loss computation
    # This is what model tries to predict during training
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    """Fine-tune BART on CNN/DailyMail.

    ==========================================================================
    COMMAND-LINE ARGUMENTS
    ==========================================================================
    --epochs: Default 3
       - Why 3? Enough for convergence, prevents overfitting
       - Full training: 3-5 epochs typical

    --batch_size: Default 4
       - Why 4? BART-large is 400M params
       - 4 fits in 8GB GPU, 8+ needs 16GB+
       - Effective batch = batch_size × gradient_accumulation

    --lr: Default 3e-5 (0.00003)
       - Why so small? Adapted weights already good
       - High LR destroys pretrained knowledge
       - 3e-5 is a good starting point

    --max_samples: Default None (all data)
       - Quick test: --max_samples 100 --epochs 1
       - Full training: no flag (uses all 287k)

    --output_dir: Default "checkpoints/bart-finetuned"
       - Where to save model checkpoints
       - Includes tokenizer and model weights

    ==========================================================================
    TRAINING PROCEDURE
    ==========================================================================
    1. Load pretrained BART (from facebook/bart-large-cnn)
    2. Load CNN/DailyMail dataset
    3. Split train/val (90/10)
    4. Preprocess (tokenize + add prefix)
    5. Configure training args
    6. Train with Seq2SeqTrainer
    7. Save model + tokenizer

    Total time: ~4-6 hours on single GPU for full dataset
    Quick test: ~5 minutes for 100 samples, 1 epoch

    ==========================================================================
    WHAT HAPPENS DURING TRAINING?
    ==========================================================================
    For each batch:
    1. Forward pass: article → model → predicted summary
    2. Loss: Compare predicted vs actual (cross-entropy)
    3. Backward pass: Compute gradients
    4. Optimizer step: Update weights to minimize loss

    Repeat for N epochs until loss converges.
    """
    parser = argparse.ArgumentParser(description="Fine-tune BART on CNN/DailyMail")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max samples for quick test"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/bart-finetuned",
        help="Output directory for checkpoints",
    )
    args = parser.parse_args()

    print(f"Device: {config.DEVICE}")
    print(f"Output dir: {args.output_dir}")

    # ==========================================================================
    # LOAD PRETRAINED MODEL
    # ==========================================================================
    # AutoTokenizer: Loads BART tokenizer (vocabulary + tokenization logic)
    # AutoModelForSeq2SeqLM: Loads BART encoder-decoder model
    # .from_pretrained() downloads from HuggingFace on first run
    print("Loading BART...")
    tokenizer = AutoTokenizer.from_pretrained(config.BART_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.BART_MODEL)

    # Move model to appropriate device (CUDA if available)
    model.to(config.DEVICE)

    # ==========================================================================
    # LOAD DATASET
    # ==========================================================================
    print("Loading dataset...")
    if args.max_samples:
        # Quick test mode: load only first N samples
        # Useful for debugging before full training
        dataset = load_dataset(
            "cnn_dailymail", "3.0.0", split=f"train[:{args.max_samples}]"
        )
    else:
        # Full training: load all 287k samples
        # WARNING: Requires significant GPU memory and time
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")

    # Split into train/validation (90/10)
    # Validation set: Used for hyperparameter tuning, prevents overfitting
    dataset = dataset.train_test_split(test_size=0.1)

    print(f"Train size: {len(dataset['train'])}")
    print(f"Val size: {len(dataset['test'])}")

    # ==========================================================================
    # PREPROCESS DATA
    # ==========================================================================
    # Tokenize all articles and summaries
    # batched=True: Process multiple samples at once (faster)
    # remove_columns: Remove original text columns (save memory)
    print("Preprocessing...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["article", "highlights", "id"],
    )

    # ==========================================================================
    # DATA COLLATOR
    # ==========================================================================
    # DataCollatorForSeq2Seq handles dynamic padding
    # Why dynamic? Different articles have different lengths
    # Collator pads to longest in batch (not fixed max)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # ==========================================================================
    # TRAINING ARGUMENTS
    # ==========================================================================
    # Seq2SeqTrainingArguments configures the training process
    # Key settings:
    # - output_dir: Where to save checkpoints
    # - num_train_epochs: How many passes through data
    # - per_device_train_batch_size: Batch size (smaller = more stable)
    # - learning_rate: How much to update weights
    # - weight_decay: L2 regularization
    # - warmup_ratio: % of steps for LR warmup
    # - logging_steps: Log progress every N steps
    # - save_steps: Save checkpoint every N steps
    # - eval_strategy: When to evaluate ("steps")
    # - save_total_limit: Keep only last N checkpoints
    # - predict_with_generate: Generate during eval (slower but accurate)
    # - fp16: Mixed precision (2x faster, less memory)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,  # Eval can use larger batch
        learning_rate=args.lr,
        weight_decay=0.01,  # L2 regularization
        warmup_ratio=0.1,  # 10% warmup steps
        logging_steps=100,  # Log every 100 steps
        save_steps=500,  # Save every 500 steps
        eval_strategy="steps",  # Evaluate during training
        save_total_limit=2,  # Keep only last 2 checkpoints
        predict_with_generate=True,  # Generate for eval metrics
        fp16=config.DEVICE == "cuda",  # Mixed precision if CUDA
        report_to="none",  # Disable wandb/tensorboard
    )

    # ==========================================================================
    # TRAINER
    # ==========================================================================
    # Seq2SeqTrainer handles training loop
    # Includes:
    # - Training/eval loop
    # - Gradient computation
    # - Checkpoint saving/loading
    # - Metric computation
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    # ==========================================================================
    # TRAIN
    # ==========================================================================
    print("Starting fine-tuning...")
    trainer.train()

    # ==========================================================================
    # SAVE
    # ==========================================================================
    # Save model and tokenizer for later use
    # Can be loaded with AbstractiveSummarizer in evaluation
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()

"""Fine-tuning script for BART summarization.

Fine-tunes facebook/bart-large-cnn on CNN/DailyMail dataset.
"""

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

import config


def preprocess_function(
    examples, tokenizer, max_source_length=1024, max_target_length=130
):
    """Preprocess for BART fine-tuning."""
    # Add prefix for summarization
    inputs = ["summarize: " + doc for doc in examples["article"]]

    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        truncation=True,
        padding="max_length",
    )

    # Tokenize targets
    labels = tokenizer(
        examples["highlights"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BART on CNN/DailyMail")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max samples for quick test"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/bart-finetuned",
        help="Output directory",
    )
    args = parser.parse_args()

    print(f"Device: {config.DEVICE}")
    print(f"Output dir: {args.output_dir}")

    # Load tokenizer and model
    print("Loading BART...")
    tokenizer = AutoTokenizer.from_pretrained(config.BART_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.BART_MODEL)
    model.to(config.DEVICE)

    # Load dataset
    print("Loading dataset...")
    if args.max_samples:
        dataset = load_dataset(
            "cnn_dailymail", "3.0.0", split=f"train[:{args.max_samples}]"
        )
    else:
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")

    # Split into train/val
    dataset = dataset.train_test_split(test_size=0.1)

    print(f"Train size: {len(dataset['train'])}")
    print(f"Val size: {len(dataset['test'])}")

    # Preprocess
    print("Preprocessing...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["article", "highlights", "id"],
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        save_steps=500,
        eval_strategy="steps",
        save_total_limit=2,
        predict_with_generate=True,
        fp16=config.DEVICE == "cuda",
        report_to="none",
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting fine-tuning...")
    trainer.train()

    # Save
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()

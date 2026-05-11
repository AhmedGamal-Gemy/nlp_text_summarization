"""Quick script to download and cache the BART model."""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Downloading BART tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
print("Tokenizer downloaded and cached")

print("Downloading BART model (1.63GB)...")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
print("Model downloaded and cached")

print("Done! Model is now cached. You can run train.py without downloading.")

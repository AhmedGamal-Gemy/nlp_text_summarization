"""FastAPI server for text summarization.

Provides REST API endpoints for both extractive and abstractive summarization.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import time

# Local imports (from src package)
from ..abstractive import AbstractiveSummarizer
from ..extractive import ExtractiveSummarizer
from ..features import TFIDFExtractor, EmbeddingScorer
from .. import config
from .. import preprocessing

app = FastAPI(
    title="Text Summarization API",
    description="Extractive and Abstractive text summarization with BART",
    version="1.0.0",
)

# Initialize models globally
_extractive_summarizer = None
_abstractive_summarizer = None
_tfidf_extractor = None


def get_extractive():
    """Lazy load extractive model."""
    global _extractive_summarizer, _tfidf_extractor
    if _extractive_summarizer is None:
        preprocessing.download_nltk_resources()
        # Fit on sample data - in production, load saved vectors
        _tfidf_extractor = TFIDFExtractor()
        # Fit on empty for now - add training data in production
        _tfidf_extractor.fit(["Sample article for initialization."])
        embedder = EmbeddingScorer()
        _extractive_summarizer = ExtractiveSummarizer(_tfidf_extractor, embedder)
    return _extractive_summarizer


def get_abstractive():
    """Lazy load abstractive model."""
    global _abstractive_summarizer
    if _abstractive_summarizer is None:
        try:
            _abstractive_summarizer = AbstractiveSummarizer()
        except Exception as e:
            raise HTTPException(f"Failed to load BART model: {e}")
    return _abstractive_summarizer


class SummarizeRequest(BaseModel):
    """Request model for summarization."""

    text: str
    model: str = "both"  # "extractive", "abstractive", or "both"
    top_k: Optional[int] = 3


class SummarizeResponse(BaseModel):
    """Response model for summarization."""

    extractive: Optional[str] = None
    abstractive: Optional[str] = None
    compression_ratio_extractive: Optional[float] = None
    compression_ratio_abstractive: Optional[float] = None
    time_extractive: Optional[float] = None
    time_abstractive: Optional[float] = None


def calc_compression(original: str, summary: str) -> float:
    """Calculate compression ratio."""
    orig_len = len(original.split())
    sum_len = len(summary.split())
    return sum_len / orig_len if orig_len > 0 else 0.0


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    print("Loading models...")
    try:
        get_extractive()
        print("Extractive model loaded")
    except Exception as e:
        print(f"Extractive load warning: {e}")

    try:
        get_abstractive()
        print("Abstractive model loaded")
    except Exception as e:
        print(f"Abstractive load warning: {e}")

    print(f"Device: {config.DEVICE}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Text Summarization API",
        "version": "1.0.0",
        "device": config.DEVICE,
        "models": {
            "extractive": "TF-IDF + sentence embeddings",
            "abstractive": "facebook/bart-large-cnn",
        },
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "device": config.DEVICE}


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Summarize text using extractive and/or abstractive models."""
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException("Text cannot be empty")

    response = SummarizeResponse()

    # Extractive
    if request.model in ["extractive", "both"]:
        start = time.time()
        try:
            extractor = get_extractive()
            response.extractive = extractor.summarize(
                request.text, request.top_k or config.TOP_K_SENTENCES
            )
            response.time_extractive = time.time() - start
            response.compression_ratio_extractive = calc_compression(
                request.text, response.extractive
            )
        except Exception as e:
            raise HTTPException(f"Extractive failed: {e}")

    # Abstractive
    if request.model in ["abstractive", "both"]:
        start = time.time()
        try:
            abstractor = get_abstractive()
            response.abstractive = abstractor.summarize(request.text)
            response.time_abstractive = time.time() - start
            response.compression_ratio_abstractive = calc_compression(
                request.text, response.abstractive
            )
        except Exception as e:
            raise HTTPException(f"Abstractive failed: {e}")

    return response


@app.get("/models")
async def models():
    """List available models."""
    return {
        "extractive": {
            "method": "TF-IDF + sentence embeddings",
            "top_k": config.TOP_K_SENTENCES,
        },
        "abstractive": {
            "model": config.BART_MODEL,
            "max_length": config.BART_MAX_LEN,
            "min_length": config.BART_MIN_LEN,
            "beams": config.BART_BEAMS,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

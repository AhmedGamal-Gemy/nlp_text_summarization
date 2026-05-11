# Services Package
"""
Business logic services for evaluation and training.
"""

from .evaluation import (
    compute_rouge,
    compression_ratio,
    evaluate_model,
    compare_models,
    qualitative_examples,
)

from .training import main as train_model
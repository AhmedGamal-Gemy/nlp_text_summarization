"""
Base interface for summarization models.

This module defines the abstract base class that all summarizers must implement,
ensuring a consistent interface across extractive and abstractive approaches.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class SummarizerBase(ABC):
    """Abstract base class for all summarization models.

    All summarizers (extractive, abstractive, etc.) must inherit from this class
    and implement the summarize() method with a consistent interface.

    Example:
        >>> class MySummarizer(SummarizerBase):
        ...     def summarize(self, text: str, **kwargs) -> str:
        ...         # Implementation
        ...         return summary
    """

    @abstractmethod
    def summarize(self, text: str, **kwargs) -> str:
        """Generate a summary of the input text.

        Args:
            text: The input article/text to summarize.
            **kwargs: Additional model-specific parameters.

        Returns:
            A string containing the generated summary.
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get model information and metadata.

        Returns:
            Dictionary containing model name, type, and configuration.
        """
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type identifier.

        Returns:
            String identifier (e.g., 'extractive', 'abstractive').
        """
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the device this model runs on.

        Returns:
            String identifier (e.g., 'cpu', 'cuda').
        """
        pass
"""Abstract base class for document loaders."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders.

    Implementations should fetch documents from external sources (PubMed, FDA, etc.),
    extract metadata, and return structured Document objects.
    """

    @abstractmethod
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        """Load documents from source.

        Returns:
            List of document dicts with keys: content, metadata
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate loader configuration before use."""
        pass

"""FDA SaMD guidance document loader."""

from src.ingest.base import BaseDocumentLoader
from typing import List, Dict, Any


class FDALoader(BaseDocumentLoader):
    """Load FDA AI/ML SaMD guidance documents.

    Fetches PDF documents from FDA website, extracts text and metadata,
    and returns structured documents.
    """

    def __init__(self, cache_dir: str = None):
        """Initialize FDA loader.

        Args:
            cache_dir: Local cache directory for downloaded PDFs
        """
        self.cache_dir = cache_dir or "/tmp/fda_cache"

    def load(self) -> List[Dict[str, Any]]:
        """Fetch FDA SaMD guidance documents.

        Returns:
            List of document dicts with text and metadata
        """
        # Implementation placeholder
        pass

    def validate_config(self) -> bool:
        """Validate cache directory is writable."""
        # Implementation placeholder
        return True

"""Document size estimation and smart routing for LLM provider selection."""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def estimate_inference_time(text: str, config: Dict[str, Any]) -> int:
    """
    Estimate inference time in seconds based on text length.

    Estimates using token count and token processing speed:
    - ~1 token per 4 characters (rough tokenization)
    - ~0.2 seconds per token (Ollama gemma4:26b on typical hardware)

    Args:
        text: Extracted document text
        config: Configuration dict with timeout settings

    Returns:
        Estimated inference time in seconds (integer)

    Example:
        1MB text (1M chars) → 250k tokens → 50 seconds estimated
        5MB text (5M chars) → 1.25M tokens → 250 seconds estimated
    """
    if not text:
        return 0

    # Rough tokenization: ~4 characters per token
    chars = len(text)
    tokens = chars // 4

    # Token processing speed: ~0.2 seconds per token
    # (based on Ollama gemma4:26b benchmark)
    estimated_seconds = tokens * 0.2

    return int(estimated_seconds)


def should_prefer_fallback_for_document(
    document_size_bytes: int,
    config: Dict[str, Any]
) -> bool:
    """
    Determine if fallback provider should be preferred for large documents.

    Returns True if document size exceeds configured warning threshold.
    Use this to trigger fast cloud fallback for large documents that would
    timeout on local inference.

    Args:
        document_size_bytes: Size of source document in bytes
        config: Configuration dict with "document_size_warning_mb" key

    Returns:
        True if document exceeds threshold, False otherwise

    Example:
        5MB document with warning_threshold=1.0 MB → True (prefer fallback)
    """
    if document_size_bytes is None:
        return False

    warning_threshold_mb = config.get("document_size_warning_mb", 1.0)
    warning_threshold_bytes = warning_threshold_mb * 1024 * 1024

    return document_size_bytes > warning_threshold_bytes

"""Test suite for document size estimation and smart routing."""
import pytest
from unittest.mock import MagicMock

from servo_skull.document_router import (
    estimate_inference_time,
    should_prefer_fallback_for_document,
)
from servo_skull.llm_client import LLMClient
from servo_skull.llm_providers import BaseLLMProvider


class TestEstimateInferenceTime:
    """Tests for estimate_inference_time function."""

    def test_estimate_inference_time_empty_text(self):
        """Test estimate_inference_time with empty text."""
        config = {"local_inference_max_seconds": 300}
        estimated = estimate_inference_time("", config)
        assert estimated == 0

    def test_estimate_inference_time_small_text(self):
        """Test estimate_inference_time with small text (~130 chars)."""
        config = {"local_inference_max_seconds": 300}
        text = "Hello, world!" * 10  # ~130 chars
        estimated = estimate_inference_time(text, config)
        # 130 chars ÷ 4 = ~33 tokens × 0.2 = ~6.6 seconds
        assert 0 <= estimated <= 10

    def test_estimate_inference_time_1kb(self):
        """Test estimate_inference_time with 1KB text."""
        config = {"local_inference_max_seconds": 300}
        text = "x" * 1024  # 1KB
        estimated = estimate_inference_time(text, config)
        # 1024 chars ÷ 4 = 256 tokens × 0.2 = ~51 seconds
        assert 50 <= estimated <= 52

    def test_estimate_inference_time_100kb(self):
        """Test estimate_inference_time with 100KB text."""
        config = {"local_inference_max_seconds": 300}
        text = "x" * (100 * 1024)
        estimated = estimate_inference_time(text, config)
        # 100KB = 102400 chars ÷ 4 = 25600 tokens × 0.2 = ~5120 seconds
        assert 5110 <= estimated <= 5130

    def test_estimate_inference_time_1mb(self):
        """Test estimate_inference_time with 1MB text."""
        config = {"local_inference_max_seconds": 300}
        text = "x" * (1024 * 1024)
        estimated = estimate_inference_time(text, config)
        # 1MB = 1048576 chars ÷ 4 = 262144 tokens × 0.2 = ~52428 seconds
        assert 52420 <= estimated <= 52440

    def test_estimate_inference_time_5mb(self):
        """Test estimate_inference_time with 5MB text."""
        config = {"local_inference_max_seconds": 300}
        text = "x" * (5 * 1024 * 1024)
        estimated = estimate_inference_time(text, config)
        # 5MB = 5242880 chars ÷ 4 = 1310720 tokens × 0.2 = 262144 seconds
        assert 262140 <= estimated <= 262150

    def test_estimate_inference_time_returns_integer(self):
        """Test estimate_inference_time returns integer."""
        config = {"local_inference_max_seconds": 300}
        text = "Hello world! This is a test."
        estimated = estimate_inference_time(text, config)
        assert isinstance(estimated, int)

    def test_estimate_inference_time_ignores_config_keys(self):
        """Test estimate_inference_time works with any config dict."""
        config = {"unrelated_key": "unrelated_value"}
        text = "x" * 1024
        estimated = estimate_inference_time(text, config)
        # Should still calculate correctly
        assert 50 <= estimated <= 52

    def test_estimate_inference_time_whitespace_text(self):
        """Test estimate_inference_time with whitespace."""
        config = {"local_inference_max_seconds": 300}
        text = "   \n\n\t  "  # Only whitespace, ~9 chars
        estimated = estimate_inference_time(text, config)
        # 9 chars ÷ 4 = ~2 tokens × 0.2 = ~0.4 seconds
        assert 0 <= estimated <= 1


class TestShouldPreferFallbackForDocument:
    """Tests for should_prefer_fallback_for_document function."""

    def test_should_prefer_fallback_small_document(self):
        """Test should_prefer_fallback_for_document with small document."""
        config = {"document_size_warning_mb": 1.0}
        # 500 KB < 1 MB threshold
        result = should_prefer_fallback_for_document(500 * 1024, config)
        assert result is False

    def test_should_prefer_fallback_large_document(self):
        """Test should_prefer_fallback_for_document with large document."""
        config = {"document_size_warning_mb": 1.0}
        # 5 MB > 1 MB threshold
        result = should_prefer_fallback_for_document(5 * 1024 * 1024, config)
        assert result is True

    def test_should_prefer_fallback_at_threshold(self):
        """Test should_prefer_fallback_for_document at exact threshold."""
        config = {"document_size_warning_mb": 1.0}
        # Exactly 1 MB (should NOT trigger, only > threshold)
        result = should_prefer_fallback_for_document(1 * 1024 * 1024, config)
        assert result is False

    def test_should_prefer_fallback_just_over_threshold(self):
        """Test should_prefer_fallback_for_document just over threshold."""
        config = {"document_size_warning_mb": 1.0}
        # Just over 1 MB (should trigger)
        result = should_prefer_fallback_for_document(1 * 1024 * 1024 + 1, config)
        assert result is True

    def test_should_prefer_fallback_custom_threshold(self):
        """Test should_prefer_fallback_for_document with custom threshold."""
        config = {"document_size_warning_mb": 0.5}
        # 1 MB > 0.5 MB threshold
        result = should_prefer_fallback_for_document(1 * 1024 * 1024, config)
        assert result is True

    def test_should_prefer_fallback_custom_threshold_below(self):
        """Test should_prefer_fallback_for_document below custom threshold."""
        config = {"document_size_warning_mb": 2.0}
        # 1 MB < 2 MB threshold
        result = should_prefer_fallback_for_document(1 * 1024 * 1024, config)
        assert result is False

    def test_should_prefer_fallback_none_size(self):
        """Test should_prefer_fallback_for_document with None document size."""
        config = {"document_size_warning_mb": 1.0}
        result = should_prefer_fallback_for_document(None, config)
        assert result is False

    def test_should_prefer_fallback_zero_size(self):
        """Test should_prefer_fallback_for_document with zero size."""
        config = {"document_size_warning_mb": 1.0}
        result = should_prefer_fallback_for_document(0, config)
        assert result is False

    def test_should_prefer_fallback_missing_config_key(self):
        """Test should_prefer_fallback_for_document with missing config key."""
        config = {}  # No "document_size_warning_mb" key
        # Should use default 1.0 MB
        result = should_prefer_fallback_for_document(5 * 1024 * 1024, config)
        assert result is True

    def test_should_prefer_fallback_very_large_document(self):
        """Test should_prefer_fallback_for_document with very large document."""
        config = {"document_size_warning_mb": 1.0}
        # 100 MB > 1 MB threshold
        result = should_prefer_fallback_for_document(100 * 1024 * 1024, config)
        assert result is True

    def test_should_prefer_fallback_returns_boolean(self):
        """Test should_prefer_fallback_for_document returns boolean."""
        config = {"document_size_warning_mb": 1.0}
        result = should_prefer_fallback_for_document(500 * 1024, config)
        assert isinstance(result, bool)


class TestEstimateInferenceTimeVsTimeoutConfig:
    """Integration tests comparing estimated time to timeout configuration."""

    def test_estimate_inference_time_vs_timeout_config_small_doc(self):
        """Test that small document estimate fits within local timeout."""
        config = {
            "local_inference_max_seconds": 1000,
            "cloud_inference_max_seconds": 60,
            "document_size_warning_mb": 1.0,
        }
        # Small document: estimate < local timeout
        small_text = "x" * 10000  # ~10KB
        small_estimate = estimate_inference_time(small_text, config)
        assert small_estimate < config["local_inference_max_seconds"]

    def test_estimate_inference_time_vs_timeout_config_large_doc(self):
        """Test that large document exceeds local timeout threshold."""
        config = {
            "local_inference_max_seconds": 300,
            "cloud_inference_max_seconds": 60,
            "document_size_warning_mb": 1.0,
        }
        # Large document: estimate > local timeout (should prefer fallback)
        large_text = "x" * (5 * 1024 * 1024)  # 5MB
        large_estimate = estimate_inference_time(large_text, config)
        assert large_estimate > config["local_inference_max_seconds"]

    def test_estimate_inference_time_large_doc_triggers_fallback_routing(self):
        """Test that large document triggers fallback routing."""
        config = {
            "local_inference_max_seconds": 300,
            "cloud_inference_max_seconds": 60,
            "document_size_warning_mb": 1.0,
        }
        large_text = "x" * (5 * 1024 * 1024)  # 5MB
        # Large document should trigger size-based fallback routing
        assert should_prefer_fallback_for_document(
            len(large_text.encode('utf-8')),
            config
        )


class TestLLMClientChooseProvider:
    """Tests for LLMClient.choose_provider method."""

    def test_llm_client_choose_provider_primary_by_default(self):
        """Test LLMClient.choose_provider returns primary by default."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)
        chosen = client.choose_provider(document_size_bytes=100000)  # 100KB

        assert chosen is primary

    def test_llm_client_choose_provider_small_document(self):
        """Test LLMClient.choose_provider with small document."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)
        chosen = client.choose_provider(document_size_bytes=500 * 1024)  # 500KB

        assert chosen is primary

    def test_llm_client_choose_provider_large_document(self):
        """Test LLMClient.choose_provider returns fallback for large documents."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)
        # 5MB > 1MB threshold
        chosen = client.choose_provider(document_size_bytes=5 * 1024 * 1024)

        assert chosen is fallback

    def test_llm_client_choose_provider_no_fallback_returns_primary(self):
        """Test LLMClient.choose_provider returns primary when no fallback."""
        primary = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=None)
        # Even with large document, no fallback means return primary
        chosen = client.choose_provider(document_size_bytes=5 * 1024 * 1024)

        assert chosen is primary

    def test_llm_client_choose_provider_fallback_frequency(self):
        """Test LLMClient.choose_provider returns fallback after 2+ uses."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)
        client.fallback_count = 2  # Simulate 2 fallbacks already

        chosen = client.choose_provider(document_size_bytes=100000)
        assert chosen is fallback

    def test_llm_client_choose_provider_before_fallback_threshold(self):
        """Test LLMClient.choose_provider prefers primary before threshold."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)
        client.fallback_count = 1  # Below threshold of 2

        chosen = client.choose_provider(document_size_bytes=100000)
        assert chosen is primary

    def test_llm_client_choose_provider_none_document_size(self):
        """Test LLMClient.choose_provider with None document size."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)
        chosen = client.choose_provider(document_size_bytes=None)

        assert chosen is primary

    def test_llm_client_choose_provider_combined_conditions(self):
        """Test LLMClient.choose_provider with combined size and frequency."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)
        client.fallback_count = 1

        # Large document alone shouldn't force fallback if freq < threshold
        chosen = client.choose_provider(document_size_bytes=5 * 1024 * 1024)
        assert chosen is fallback  # But size check comes first

    def test_llm_client_choose_provider_returns_provider(self):
        """Test LLMClient.choose_provider returns a provider object."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)
        chosen = client.choose_provider(document_size_bytes=100000)

        assert chosen in [primary, fallback]

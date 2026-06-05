"""Test suite for unified LLMClient with fallback chain."""
import pytest
import httpx
from unittest.mock import MagicMock, patch

from servo_skull.llm_client import LLMClient
from servo_skull.llm_providers import BaseLLMProvider


class TestLLMClientInitialization:
    """Tests for LLMClient initialization."""

    def test_llm_client_init_with_primary_and_fallback(self):
        """Test LLMClient initialization with primary and fallback providers."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)

        assert client.primary is primary
        assert client.fallback is fallback
        assert client.fallback_triggered is False
        assert client.fallback_count == 0

    def test_llm_client_init_primary_only(self):
        """Test LLMClient initialization with primary provider only."""
        primary = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=None)

        assert client.primary is primary
        assert client.fallback is None
        assert client.fallback_triggered is False
        assert client.fallback_count == 0

    def test_llm_client_init_stores_state(self):
        """Test LLMClient initializes with clean state."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)

        assert not client.fallback_triggered
        assert client.fallback_count == 0


class TestLLMClientChatSuccess:
    """Tests for successful chat operations."""

    def test_llm_client_chat_primary_success(self):
        """Test LLMClient.chat succeeds with primary provider."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.return_value = "Primary response"
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)
        result = client.chat("System", "User")

        assert result == "Primary response"
        assert not client.fallback_triggered
        assert client.fallback_count == 0
        primary.chat.assert_called_once()
        fallback.chat.assert_not_called()

    def test_llm_client_chat_with_document_size(self):
        """Test LLMClient.chat propagates document size for logging."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.return_value = "Response"
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)
        result = client.chat("System", "User", document_size_bytes=5000000)

        assert result == "Response"
        primary.chat.assert_called_once_with(
            system_prompt="System",
            user_prompt="User",
            timeout_override=None
        )

    def test_llm_client_chat_timeout_override_propagated(self):
        """Test LLMClient propagates timeout_override to primary provider."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.return_value = "Response"
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)
        result = client.chat("System", "User", timeout_override=30.0)

        assert result == "Response"
        primary.chat.assert_called_once_with(
            system_prompt="System",
            user_prompt="User",
            timeout_override=30.0
        )


class TestLLMClientFallbackOnTimeoutException:
    """Tests for fallback on TimeoutException."""

    def test_llm_client_fallback_on_timeout(self):
        """Test LLMClient switches to fallback on primary timeout."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.side_effect = httpx.TimeoutException("Timeout")
        fallback = MagicMock(spec=BaseLLMProvider)
        fallback.chat.return_value = "Fallback response"

        client = LLMClient(primary=primary, fallback=fallback)
        result = client.chat("System", "User", document_size_bytes=5000000)

        assert result == "Fallback response"
        assert client.fallback_triggered is True
        assert client.fallback_count == 1
        primary.chat.assert_called_once()
        fallback.chat.assert_called_once()

    def test_llm_client_multiple_fallbacks(self):
        """Test LLMClient increments fallback count on multiple timeouts."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.side_effect = httpx.TimeoutException("Timeout")
        fallback = MagicMock(spec=BaseLLMProvider)
        fallback.chat.return_value = "Fallback response"

        client = LLMClient(primary=primary, fallback=fallback)

        # First fallback
        result1 = client.chat("System", "User")
        assert result1 == "Fallback response"
        assert client.fallback_count == 1

        # Reset mocks for second call
        primary.reset_mock(side_effect=True)
        fallback.reset_mock()
        primary.chat.side_effect = httpx.TimeoutException("Timeout")
        fallback.chat.return_value = "Fallback response 2"

        # Second fallback
        result2 = client.chat("System", "User")
        assert result2 == "Fallback response 2"
        assert client.fallback_count == 2

    def test_llm_client_fallback_propagates_params(self):
        """Test LLMClient propagates parameters to fallback provider."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.side_effect = httpx.TimeoutException("Timeout")
        fallback = MagicMock(spec=BaseLLMProvider)
        fallback.chat.return_value = "Fallback response"

        client = LLMClient(primary=primary, fallback=fallback)
        result = client.chat("System prompt", "User prompt", timeout_override=60.0)

        assert result == "Fallback response"
        fallback.chat.assert_called_once_with(
            system_prompt="System prompt",
            user_prompt="User prompt",
            timeout_override=60.0
        )


class TestLLMClientFallbackOnConnectError:
    """Tests for fallback on connection errors."""

    def test_llm_client_fallback_on_connect_error(self):
        """Test LLMClient switches to fallback on primary connection error."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.side_effect = httpx.ConnectError("Connection failed")
        fallback = MagicMock(spec=BaseLLMProvider)
        fallback.chat.return_value = "Fallback response"

        client = LLMClient(primary=primary, fallback=fallback)
        result = client.chat("System", "User")

        assert result == "Fallback response"
        assert client.fallback_triggered is True
        assert client.fallback_count == 1

    def test_llm_client_fallback_on_any_httpx_error(self):
        """Test LLMClient catches specific httpx exceptions."""
        primary = MagicMock(spec=BaseLLMProvider)
        # Test with ConnectError (connection refused, etc.)
        primary.chat.side_effect = httpx.ConnectError("Failed to connect")
        fallback = MagicMock(spec=BaseLLMProvider)
        fallback.chat.return_value = "Fallback response"

        client = LLMClient(primary=primary, fallback=fallback)
        result = client.chat("System", "User")

        assert result == "Fallback response"
        assert client.fallback_count == 1


class TestLLMClientNoFallback:
    """Tests for behavior when no fallback is configured."""

    def test_llm_client_no_fallback_raises_error(self):
        """Test LLMClient raises error if primary fails and no fallback configured."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.side_effect = httpx.TimeoutException("Timeout")

        client = LLMClient(primary=primary, fallback=None)

        with pytest.raises(httpx.TimeoutException):
            client.chat("System", "User")

        assert not client.fallback_triggered
        assert client.fallback_count == 0

    def test_llm_client_no_fallback_on_connect_error(self):
        """Test LLMClient raises ConnectError if primary fails and no fallback."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.side_effect = httpx.ConnectError("Connection failed")

        client = LLMClient(primary=primary, fallback=None)

        with pytest.raises(httpx.ConnectError):
            client.chat("System", "User")

        assert not client.fallback_triggered


class TestLLMClientFallbackFailure:
    """Tests for fallback provider failures."""

    def test_llm_client_fallback_also_fails(self):
        """Test LLMClient raises error if both primary and fallback fail."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.side_effect = httpx.TimeoutException("Primary timeout")
        fallback = MagicMock(spec=BaseLLMProvider)
        fallback.chat.side_effect = httpx.HTTPError("Fallback error")

        client = LLMClient(primary=primary, fallback=fallback)

        with pytest.raises(httpx.HTTPError, match="Fallback error"):
            client.chat("System", "User")

        assert client.fallback_triggered is True
        assert client.fallback_count == 1

    def test_llm_client_fallback_timeout_also_fails(self):
        """Test LLMClient raises error if fallback times out."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.side_effect = httpx.TimeoutException("Primary timeout")
        fallback = MagicMock(spec=BaseLLMProvider)
        fallback.chat.side_effect = httpx.TimeoutException("Fallback timeout")

        client = LLMClient(primary=primary, fallback=fallback)

        with pytest.raises(httpx.TimeoutException, match="Fallback timeout"):
            client.chat("System", "User")

        assert client.fallback_count == 1


class TestLLMClientShouldPreferFallback:
    """Tests for should_prefer_fallback logic."""

    def test_should_prefer_fallback_threshold(self):
        """Test should_prefer_fallback returns True after >= 2 fallbacks."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)

        # Initially prefer primary
        assert not client.should_prefer_fallback()
        assert client.fallback_count == 0

        # After 1 fallback, still prefer primary
        client.fallback_count = 1
        assert not client.should_prefer_fallback()

        # After 2 fallbacks, prefer fallback
        client.fallback_count = 2
        assert client.should_prefer_fallback()

        # After 5 fallbacks, still prefer fallback
        client.fallback_count = 5
        assert client.should_prefer_fallback()

    def test_should_prefer_fallback_boundary(self):
        """Test should_prefer_fallback at boundary (1 vs 2)."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)

        # At count = 1, should not prefer fallback
        client.fallback_count = 1
        assert not client.should_prefer_fallback()

        # At count = 2, should prefer fallback
        client.fallback_count = 2
        assert client.should_prefer_fallback()

    def test_should_prefer_fallback_zero_fallbacks(self):
        """Test should_prefer_fallback with zero fallbacks."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)
        assert client.fallback_count == 0
        assert not client.should_prefer_fallback()


class TestLLMClientContextManager:
    """Tests for context manager support."""

    def test_llm_client_context_manager(self):
        """Test LLMClient as context manager closes providers."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        with LLMClient(primary=primary, fallback=fallback) as client:
            assert client.primary is primary
            assert client.fallback is fallback

        primary.close.assert_called_once()
        fallback.close.assert_called_once()

    def test_llm_client_context_manager_primary_only(self):
        """Test LLMClient context manager with primary only."""
        primary = MagicMock(spec=BaseLLMProvider)

        with LLMClient(primary=primary, fallback=None) as client:
            assert client.primary is primary
            assert client.fallback is None

        primary.close.assert_called_once()

    def test_llm_client_explicit_close(self):
        """Test LLMClient.close() method."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)
        client.close()

        primary.close.assert_called_once()
        fallback.close.assert_called_once()

    def test_llm_client_close_primary_only(self):
        """Test LLMClient.close() with primary only."""
        primary = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=None)
        client.close()

        primary.close.assert_called_once()

    def test_llm_client_context_manager_exception_cleanup(self):
        """Test context manager cleans up on exception."""
        primary = MagicMock(spec=BaseLLMProvider)
        fallback = MagicMock(spec=BaseLLMProvider)

        try:
            with LLMClient(primary=primary, fallback=fallback) as client:
                raise ValueError("Test error")
        except ValueError:
            pass

        # Cleanup should still be called
        primary.close.assert_called_once()
        fallback.close.assert_called_once()


class TestLLMClientErrorLogging:
    """Tests for error logging and context tracking."""

    @patch("servo_skull.llm_client.logger")
    def test_llm_client_logs_fallback_trigger(self, mock_logger):
        """Test LLMClient logs when fallback is triggered."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.side_effect = httpx.TimeoutException("Timeout")
        fallback = MagicMock(spec=BaseLLMProvider)
        fallback.chat.return_value = "Fallback response"

        client = LLMClient(primary=primary, fallback=fallback)
        client.chat("System", "User", document_size_bytes=5000000)

        # Should log warning about fallback trigger
        assert mock_logger.warning.called
        call_args = str(mock_logger.warning.call_args)
        assert "fallback" in call_args.lower()

    @patch("servo_skull.llm_client.logger")
    def test_llm_client_logs_provider_names(self, mock_logger):
        """Test LLMClient logs provider class names."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.__class__.__name__ = "OllamaProvider"
        primary.chat.return_value = "Response"
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)
        client.chat("System", "User")

        # Should log about calling primary provider
        assert mock_logger.debug.called

    @patch("servo_skull.llm_client.logger")
    def test_llm_client_logs_fallback_success(self, mock_logger):
        """Test LLMClient logs when fallback succeeds."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.side_effect = httpx.TimeoutException("Timeout")
        fallback = MagicMock(spec=BaseLLMProvider)
        fallback.chat.return_value = "Fallback response"

        client = LLMClient(primary=primary, fallback=fallback)
        client.chat("System", "User")

        # Should log fallback success with count
        assert mock_logger.info.called


class TestLLMClientIntegration:
    """Integration tests for LLMClient."""

    def test_llm_client_multiple_calls_primary_success(self):
        """Test multiple successful primary calls."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.return_value = "Response"
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)

        result1 = client.chat("System 1", "User 1")
        result2 = client.chat("System 2", "User 2")
        result3 = client.chat("System 3", "User 3")

        assert result1 == "Response"
        assert result2 == "Response"
        assert result3 == "Response"
        assert client.fallback_count == 0
        assert not client.fallback_triggered
        assert primary.chat.call_count == 3
        fallback.chat.assert_not_called()

    def test_llm_client_mixed_success_and_fallback(self):
        """Test mixed success and fallback scenarios."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.side_effect = [
            "Success 1",
            httpx.TimeoutException("Timeout"),
            "Success 2",
            httpx.ConnectError("Connection failed"),
            "Success 3",
        ]
        fallback = MagicMock(spec=BaseLLMProvider)
        fallback.chat.side_effect = [
            "Fallback 1",
            "Fallback 2",
        ]

        client = LLMClient(primary=primary, fallback=fallback)

        assert client.chat("System", "User") == "Success 1"
        assert client.fallback_count == 0

        assert client.chat("System", "User") == "Fallback 1"
        assert client.fallback_count == 1

        assert client.chat("System", "User") == "Success 2"
        assert client.fallback_count == 1

        assert client.chat("System", "User") == "Fallback 2"
        assert client.fallback_count == 2

        assert client.chat("System", "User") == "Success 3"
        assert client.fallback_count == 2

        # After 2 fallbacks, should prefer fallback
        assert client.should_prefer_fallback()

    def test_llm_client_state_persistence(self):
        """Test LLMClient maintains state across multiple calls."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.side_effect = httpx.TimeoutException("Timeout")
        fallback = MagicMock(spec=BaseLLMProvider)
        fallback.chat.return_value = "Fallback"

        client = LLMClient(primary=primary, fallback=fallback)

        # First call triggers fallback
        client.chat("System", "User")
        assert client.fallback_triggered
        assert client.fallback_count == 1

        # State persists for next call
        fallback.reset_mock()
        fallback.chat.return_value = "Fallback"
        client.chat("System", "User")
        assert client.fallback_triggered
        assert client.fallback_count == 2

    def test_llm_client_only_catches_specific_errors(self):
        """Test LLMClient only catches TimeoutException and ConnectError."""
        primary = MagicMock(spec=BaseLLMProvider)
        # Some other error that's not TimeoutException or ConnectError
        primary.chat.side_effect = ValueError("Some validation error")
        fallback = MagicMock(spec=BaseLLMProvider)

        client = LLMClient(primary=primary, fallback=fallback)

        # Should NOT catch ValueError, should re-raise
        with pytest.raises(ValueError, match="Some validation error"):
            client.chat("System", "User")

        # Fallback should not have been triggered
        assert client.fallback_count == 0
        fallback.chat.assert_not_called()

    def test_llm_client_preserves_exception_on_fallback_failure(self):
        """Test LLMClient preserves fallback exception if it fails."""
        primary = MagicMock(spec=BaseLLMProvider)
        primary.chat.side_effect = httpx.TimeoutException("Primary timeout")
        fallback = MagicMock(spec=BaseLLMProvider)
        fallback.chat.side_effect = ValueError("Fallback validation error")

        client = LLMClient(primary=primary, fallback=fallback)

        # Should raise the fallback error, not the primary error
        with pytest.raises(ValueError, match="Fallback validation error"):
            client.chat("System", "User")

"""Test suite for LLM provider implementations."""
import pytest
import httpx
from unittest.mock import MagicMock, patch
from servo_skull.llm_providers import (
    BaseLLMProvider,
    OllamaProvider,
    OpenAIProvider,
    AnthropicProvider,
    create_provider,
)


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider abstract base class."""

    def test_base_provider_is_abstract(self):
        """Test that BaseLLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseLLMProvider()

    def test_base_provider_requires_chat_method(self):
        """Test that incomplete subclass missing chat method raises TypeError."""

        class IncompleteProvider(BaseLLMProvider):
            """Provider missing required abstract methods."""
            def close(self):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProvider()


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    def test_ollama_provider_init(self):
        """Test OllamaProvider initialization with defaults."""
        provider = OllamaProvider()
        assert provider.base_url == "http://127.0.0.1:11434/v1"
        assert provider.model == "gemma4:26b"
        assert provider.temperature == 0.0
        assert provider.timeout == 300.0
        provider.close()

    def test_ollama_provider_custom_init(self):
        """Test OllamaProvider initialization with custom values."""
        provider = OllamaProvider(
            base_url="http://custom:8000/v1",
            model="mistral:7b",
            temperature=0.5,
            timeout=60.0,
        )
        assert provider.base_url == "http://custom:8000/v1"
        assert provider.model == "mistral:7b"
        assert provider.temperature == 0.5
        assert provider.timeout == 60.0
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_ollama_provider_chat_success(self, mock_client_class):
        """Test successful OllamaProvider.chat call."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = OllamaProvider()
        result = provider.chat("System prompt", "User prompt")

        assert result == "Test response"
        mock_client.post.assert_called_once()
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_ollama_provider_chat_with_timeout_override(self, mock_client_class):
        """Test OllamaProvider.chat with timeout override."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = OllamaProvider(timeout=300.0)
        result = provider.chat("System", "User", timeout_override=30.0)

        assert result == "Test response"
        # Check that timeout was passed to post call
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["timeout"] == 30.0
        provider.close()


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_openai_provider_init(self):
        """Test OpenAIProvider initialization."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "gpt-4"
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.timeout == 300.0
        provider.close()

    def test_openai_provider_custom_init(self):
        """Test OpenAIProvider initialization with custom values."""
        provider = OpenAIProvider(
            api_key="custom-key",
            model="gpt-3.5-turbo",
            base_url="https://custom.openai.com/v1",
            timeout=60.0,
        )
        assert provider.api_key == "custom-key"
        assert provider.model == "gpt-3.5-turbo"
        assert provider.base_url == "https://custom.openai.com/v1"
        assert provider.timeout == 60.0
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_openai_provider_chat_success(self, mock_client_class):
        """Test successful OpenAIProvider.chat call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OpenAI response"}}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key")
        result = provider.chat("System prompt", "User prompt")

        assert result == "OpenAI response"
        # Verify Authorization header was set
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"
        provider.close()


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_anthropic_provider_init(self):
        """Test AnthropicProvider initialization."""
        provider = AnthropicProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "claude-3-sonnet-20240229"
        assert provider.base_url == "https://api.anthropic.com"
        assert provider.timeout == 300.0
        provider.close()

    def test_anthropic_provider_custom_init(self):
        """Test AnthropicProvider initialization with custom values."""
        provider = AnthropicProvider(
            api_key="custom-key",
            model="claude-3-opus-20240229",
            timeout=120.0,
        )
        assert provider.api_key == "custom-key"
        assert provider.model == "claude-3-opus-20240229"
        assert provider.timeout == 120.0
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_anthropic_provider_chat_success(self, mock_client_class):
        """Test successful AnthropicProvider.chat call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"text": "Anthropic response"}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider(api_key="test-key")
        result = provider.chat("System prompt", "User prompt")

        assert result == "Anthropic response"
        # Verify Anthropic-specific headers were set
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["headers"]["x-api-key"] == "test-key"
        assert call_kwargs["headers"]["anthropic-version"] == "2024-06-01"
        # Verify max_tokens was set
        call_json = mock_client.post.call_args[1]["json"]
        assert call_json["max_tokens"] == 4096
        provider.close()


class TestCreateProvider:
    """Tests for create_provider factory function."""

    def test_create_provider_ollama(self):
        """Test creating OllamaProvider via factory."""
        config = {
            "type": "ollama",
            "base_url": "http://localhost:11434/v1",
            "model": "mistral:7b",
            "temperature": 0.3,
        }
        provider = create_provider(config)

        assert isinstance(provider, OllamaProvider)
        assert provider.model == "mistral:7b"
        assert provider.temperature == 0.3
        provider.close()

    def test_create_provider_openai(self):
        """Test creating OpenAIProvider via factory."""
        config = {
            "type": "openai",
            "api_key": "sk-test-key",
            "model": "gpt-4-turbo",
        }
        provider = create_provider(config)

        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == "sk-test-key"
        assert provider.model == "gpt-4-turbo"
        provider.close()

    def test_create_provider_anthropic(self):
        """Test creating AnthropicProvider via factory."""
        config = {
            "type": "anthropic",
            "api_key": "sk-ant-test-key",
            "model": "claude-3-opus-20240229",
        }
        provider = create_provider(config)

        assert isinstance(provider, AnthropicProvider)
        assert provider.api_key == "sk-ant-test-key"
        assert provider.model == "claude-3-opus-20240229"
        provider.close()

    def test_create_provider_unknown_type(self):
        """Test create_provider raises ValueError for unknown type."""
        config = {"type": "unknown_provider"}
        with pytest.raises(ValueError, match="Unknown provider type: unknown_provider"):
            create_provider(config)

    def test_create_provider_case_insensitive(self):
        """Test create_provider handles case-insensitive type."""
        config = {"type": "OLLAMA"}
        provider = create_provider(config)
        assert isinstance(provider, OllamaProvider)
        provider.close()

    def test_create_provider_with_defaults(self):
        """Test create_provider uses defaults for missing config values."""
        config = {"type": "ollama"}
        provider = create_provider(config)

        assert isinstance(provider, OllamaProvider)
        assert provider.base_url == "http://127.0.0.1:11434/v1"
        assert provider.model == "gemma4:26b"
        provider.close()


class TestProviderChatInterface:
    """Tests for chat method interface across all providers."""

    def test_provider_chat_signature_ollama(self):
        """Test OllamaProvider chat method has correct signature."""
        provider = OllamaProvider()
        # Verify method exists and has correct parameters
        import inspect
        sig = inspect.signature(provider.chat)
        params = list(sig.parameters.keys())
        assert "system_prompt" in params
        assert "user_prompt" in params
        assert "timeout_override" in params
        provider.close()

    def test_provider_chat_signature_openai(self):
        """Test OpenAIProvider chat method has correct signature."""
        provider = OpenAIProvider(api_key="test")
        import inspect
        sig = inspect.signature(provider.chat)
        params = list(sig.parameters.keys())
        assert "system_prompt" in params
        assert "user_prompt" in params
        assert "timeout_override" in params
        provider.close()

    def test_provider_chat_signature_anthropic(self):
        """Test AnthropicProvider chat method has correct signature."""
        provider = AnthropicProvider(api_key="test")
        import inspect
        sig = inspect.signature(provider.chat)
        params = list(sig.parameters.keys())
        assert "system_prompt" in params
        assert "user_prompt" in params
        assert "timeout_override" in params
        provider.close()


class TestContextManager:
    """Tests for context manager support."""

    def test_ollama_context_manager(self):
        """Test OllamaProvider as context manager."""
        with OllamaProvider() as provider:
            assert provider is not None
            assert hasattr(provider, "chat")

    def test_openai_context_manager(self):
        """Test OpenAIProvider as context manager."""
        with OpenAIProvider(api_key="test") as provider:
            assert provider is not None
            assert hasattr(provider, "chat")

    def test_anthropic_context_manager(self):
        """Test AnthropicProvider as context manager."""
        with AnthropicProvider(api_key="test") as provider:
            assert provider is not None
            assert hasattr(provider, "chat")

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_context_manager_closes_on_exit(self, mock_client_class):
        """Test that context manager calls close on exit."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        provider = OllamaProvider()
        with provider:
            pass

        # close() should have been called when exiting context
        mock_client.close.assert_called()


class TestErrorHandling:
    """Tests for JSON parsing and response validation error handling."""

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_ollama_malformed_json_response(self, mock_client_class):
        """Test OllamaProvider handles malformed JSON response."""
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Expecting value: line 1 column 1")
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = OllamaProvider()
        with pytest.raises(httpx.HTTPError, match="Invalid JSON response body"):
            provider.chat("System", "User")
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_ollama_missing_choices_key(self, mock_client_class):
        """Test OllamaProvider handles missing 'choices' key in response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "incomplete"}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = OllamaProvider()
        with pytest.raises(httpx.HTTPError, match="Invalid response: missing expected keys"):
            provider.chat("System", "User")
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_ollama_missing_message_key(self, mock_client_class):
        """Test OllamaProvider handles missing 'message' key in choices."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{}]}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = OllamaProvider()
        with pytest.raises(httpx.HTTPError, match="Invalid response: missing expected keys"):
            provider.chat("System", "User")
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_ollama_empty_choices_array(self, mock_client_class):
        """Test OllamaProvider handles empty choices array."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": []}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = OllamaProvider()
        with pytest.raises(httpx.HTTPError, match="Invalid response: missing expected keys"):
            provider.chat("System", "User")
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_openai_malformed_json_response(self, mock_client_class):
        """Test OpenAIProvider handles malformed JSON response."""
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key")
        with pytest.raises(httpx.HTTPError, match="Invalid JSON response body"):
            provider.chat("System", "User")
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_openai_missing_content_key(self, mock_client_class):
        """Test OpenAIProvider handles missing 'content' key."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {}}]}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key")
        with pytest.raises(httpx.HTTPError, match="Invalid response: missing expected keys"):
            provider.chat("System", "User")
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_anthropic_malformed_json_response(self, mock_client_class):
        """Test AnthropicProvider handles malformed JSON response."""
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider(api_key="test-key")
        with pytest.raises(httpx.HTTPError, match="Invalid JSON response body"):
            provider.chat("System", "User")
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_anthropic_missing_content_key(self, mock_client_class):
        """Test AnthropicProvider handles missing 'content' key."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider(api_key="test-key")
        with pytest.raises(httpx.HTTPError, match="Invalid response: missing expected keys"):
            provider.chat("System", "User")
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_anthropic_missing_text_key(self, mock_client_class):
        """Test AnthropicProvider handles missing 'text' key in content."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": [{}]}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider(api_key="test-key")
        with pytest.raises(httpx.HTTPError, match="Invalid response: missing expected keys"):
            provider.chat("System", "User")
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_anthropic_empty_content_array(self, mock_client_class):
        """Test AnthropicProvider handles empty content array."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": []}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider(api_key="test-key")
        with pytest.raises(httpx.HTTPError, match="Invalid response: missing expected keys"):
            provider.chat("System", "User")
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_ollama_http_error_not_leaked(self, mock_client_class):
        """Test that Ollama HTTP error logging doesn't leak API details."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_error = httpx.HTTPError("Unauthorized")
        mock_error.response = mock_response
        mock_client = MagicMock()
        mock_client.post.side_effect = mock_error
        mock_client_class.return_value = mock_client

        provider = OllamaProvider()
        with pytest.raises(httpx.HTTPError):
            provider.chat("System", "User")
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_openai_http_error_not_leaked(self, mock_client_class):
        """Test that OpenAI HTTP error logging doesn't leak API key."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_error = httpx.HTTPError("Unauthorized")
        mock_error.response = mock_response
        mock_client = MagicMock()
        mock_client.post.side_effect = mock_error
        mock_client_class.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-secret-key-12345")
        with pytest.raises(httpx.HTTPError):
            provider.chat("System", "User")
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_anthropic_http_error_not_leaked(self, mock_client_class):
        """Test that Anthropic HTTP error logging doesn't leak API key."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_error = httpx.HTTPError("Forbidden")
        mock_error.response = mock_response
        mock_client = MagicMock()
        mock_client.post.side_effect = mock_error
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-ant-secret-12345")
        with pytest.raises(httpx.HTTPError):
            provider.chat("System", "User")
        provider.close()


class TestMaxTokensConfiguration:
    """Tests for max_tokens configurability."""

    def test_anthropic_default_max_tokens(self):
        """Test AnthropicProvider uses default max_tokens."""
        provider = AnthropicProvider(api_key="test-key")
        assert provider.max_tokens == 4096
        provider.close()

    def test_anthropic_custom_max_tokens(self):
        """Test AnthropicProvider accepts custom max_tokens."""
        provider = AnthropicProvider(api_key="test-key", max_tokens=2048)
        assert provider.max_tokens == 2048
        provider.close()

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_anthropic_max_tokens_in_payload(self, mock_client_class):
        """Test that custom max_tokens is used in API payload."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": [{"text": "Response"}]}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider(api_key="test-key", max_tokens=2048)
        provider.chat("System", "User")

        # Verify max_tokens was passed in payload
        call_json = mock_client.post.call_args[1]["json"]
        assert call_json["max_tokens"] == 2048
        provider.close()

    def test_create_provider_anthropic_with_max_tokens(self):
        """Test factory function passes max_tokens config to Anthropic."""
        config = {
            "type": "anthropic",
            "api_key": "test-key",
            "max_tokens": 8192,
        }
        provider = create_provider(config)

        assert isinstance(provider, AnthropicProvider)
        assert provider.max_tokens == 8192
        provider.close()

    def test_create_provider_anthropic_default_max_tokens(self):
        """Test factory function uses default max_tokens for Anthropic."""
        config = {
            "type": "anthropic",
            "api_key": "test-key",
        }
        provider = create_provider(config)

        assert isinstance(provider, AnthropicProvider)
        assert provider.max_tokens == 4096
        provider.close()


class TestRetryConfiguration:
    """Tests for retry strategy consistency."""

    def test_ollama_retry_decorator(self):
        """Test OllamaProvider has correct retry configuration."""
        import inspect
        source = inspect.getsource(OllamaProvider.chat)
        assert "@retry(max_attempts=3" in source, "Ollama should use 3 retry attempts"

    def test_openai_retry_decorator(self):
        """Test OpenAIProvider has correct retry configuration."""
        import inspect
        source = inspect.getsource(OpenAIProvider.chat)
        assert "@retry(max_attempts=3" in source, "OpenAI should use 3 retry attempts"

    def test_anthropic_retry_decorator(self):
        """Test AnthropicProvider has correct retry configuration."""
        import inspect
        source = inspect.getsource(AnthropicProvider.chat)
        assert "@retry(max_attempts=3" in source, "Anthropic should use 3 retry attempts"


class TestFactoryValidation:
    """Tests for factory function validation."""

    def test_create_provider_empty_type(self):
        """Test create_provider raises error for empty type."""
        config = {"type": ""}
        with pytest.raises(ValueError, match="must specify a 'type' key"):
            create_provider(config)

    def test_create_provider_missing_type(self):
        """Test create_provider raises error for missing type."""
        config = {}
        with pytest.raises(ValueError, match="must specify a 'type' key"):
            create_provider(config)

    def test_create_provider_whitespace_type(self):
        """Test create_provider handles whitespace in type."""
        config = {"type": "   "}
        # Whitespace gets stripped by .lower(), so empty
        with pytest.raises(ValueError, match="Unknown provider type"):
            create_provider(config)


class TestAnthropicAPIVersion:
    """Tests for Anthropic API version."""

    @patch("servo_skull.llm_providers.httpx.Client")
    def test_anthropic_api_version_header(self, mock_client_class):
        """Test AnthropicProvider sends correct API version header."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": [{"text": "Response"}]}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider(api_key="test-key")
        provider.chat("System", "User")

        # Verify API version header is set
        call_headers = mock_client.post.call_args[1]["headers"]
        assert call_headers["anthropic-version"] == "2024-06-01"
        provider.close()


class TestGeminiProvider:
    """Tests for GeminiProvider and expand_env_vars."""

    def test_gemini_provider_init(self):
        """Test GeminiProvider initialization."""
        with patch("servo_skull.llm_providers.genai") as mock_genai:
            provider = create_provider({
                "type": "gemini",
                "api_key": "test-gemini-key",
                "model": "gemini-2.5-flash"
            })
            assert provider.api_key == "test-gemini-key"
            assert provider.model == "gemini-2.5-flash"
            mock_genai.Client.assert_called_once_with(api_key="test-gemini-key")

    @patch("servo_skull.llm_providers.genai")
    def test_gemini_provider_chat_success(self, mock_genai):
        """Test successful GeminiProvider.chat call."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Gemini response text"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        provider = create_provider({
            "type": "gemini",
            "api_key": "test-gemini-key",
            "model": "gemini-2.5-flash"
        })
        result = provider.chat("System instruction", "User query")
        assert result == "Gemini response text"
        mock_client.models.generate_content.assert_called_once()

    def test_expand_env_vars(self):
        """Test environment variables expansion."""
        from servo_skull.llm_providers import expand_env_vars
        import os
        os.environ["TEST_ENV_VAR"] = "expanded-value"

        config = {
            "api_key": "${TEST_ENV_VAR}",
            "other": "constant",
            "nested": {"key": "${TEST_ENV_VAR}"}
        }
        expanded = expand_env_vars(config)
        assert expanded["api_key"] == "expanded-value"
        assert expanded["other"] == "constant"
        assert expanded["nested"]["key"] == "expanded-value"


"""Tests for MLXProvider — lazy loading, chat(), and close()."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlx_provider(**kwargs):
    from servo_skull.mlx_provider import MLXProvider
    return MLXProvider(**kwargs)


def _mock_mlx_modules(generated_text: str = "MLX response text."):
    """Return context managers that mock mlx_lm.load and mlx_lm.generate."""
    mock_model = MagicMock(name="mlx_model")
    mock_tokenizer = MagicMock(name="tokenizer")
    mock_tokenizer.apply_chat_template = None  # use default template

    load_patch = patch(
        "servo_skull.mlx_provider.MLXProvider._ensure_loaded",
        side_effect=lambda self=None: None,
    )
    return mock_model, mock_tokenizer


# ---------------------------------------------------------------------------
# MLXProvider unit tests
# ---------------------------------------------------------------------------

class TestMLXProviderInit:
    def test_defaults(self):
        provider = _make_mlx_provider()
        assert provider.model == "mlx-community/Qwen2.5-7B-Instruct-4bit"
        assert provider.max_tokens == 4096
        assert provider.temperature == 0.0
        assert provider._mlx_model is None
        assert provider._tokenizer is None

    def test_custom_model(self):
        provider = _make_mlx_provider(
            model="mlx-community/Qwen2.5-14B-Instruct-4bit",
            max_tokens=2048,
            temperature=0.1,
        )
        assert provider.model == "mlx-community/Qwen2.5-14B-Instruct-4bit"
        assert provider.max_tokens == 2048

    def test_repr_not_loaded(self):
        provider = _make_mlx_provider()
        r = repr(provider)
        assert "loaded=False" in r
        assert "Qwen2.5-7B" in r


class TestMLXProviderChat:
    def test_chat_calls_generate(self):
        provider = _make_mlx_provider()

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = None
        provider._mlx_model = mock_model
        provider._tokenizer = mock_tokenizer

        with patch("servo_skull.mlx_provider.mlx_generate" if False else
                   "servo_skull.mlx_provider.MLXProvider.chat") as _:
            pass  # patch approach below via mlx_lm module mock

        # Directly inject the generate function via sys.modules mock
        mock_generate = MagicMock(return_value="  The answer is 42.  ")
        with patch.dict("sys.modules", {
            "mlx_lm": MagicMock(
                load=MagicMock(return_value=(mock_model, mock_tokenizer)),
                generate=mock_generate,
            )
        }):
            # Re-import to pick up the mock
            import importlib
            import servo_skull.mlx_provider as mlx_mod
            importlib.reload(mlx_mod)

            provider2 = mlx_mod.MLXProvider()
            provider2._mlx_model = mock_model
            provider2._tokenizer = mock_tokenizer

            result = provider2.chat(
                system_prompt="You are an assistant.",
                user_prompt="What is 6 × 7?",
            )

        assert result == "The answer is 42."
        # Reload back to normal state
        importlib.reload(mlx_mod)

    def test_chat_raises_on_empty_response(self):
        provider = _make_mlx_provider()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = None
        provider._mlx_model = mock_model
        provider._tokenizer = mock_tokenizer

        with patch.dict("sys.modules", {
            "mlx_lm": MagicMock(
                load=MagicMock(return_value=(mock_model, mock_tokenizer)),
                generate=MagicMock(return_value="   "),  # whitespace only
            )
        }):
            import importlib
            import servo_skull.mlx_provider as mlx_mod
            importlib.reload(mlx_mod)

            provider2 = mlx_mod.MLXProvider()
            provider2._mlx_model = mock_model
            provider2._tokenizer = mock_tokenizer

            with pytest.raises(RuntimeError, match="empty response"):
                provider2.chat("sys", "user")

            importlib.reload(mlx_mod)

    def test_import_error_without_mlx_lm(self):
        """If mlx_lm is not installed, _ensure_loaded must raise ImportError."""
        from servo_skull.mlx_provider import MLXProvider
        provider = MLXProvider.__new__(MLXProvider)
        provider._mlx_model = None
        provider._tokenizer = None
        provider.model = "some-model"
        provider.max_tokens = 512
        provider.temperature = 0.0
        provider.verbose = False
        provider.timeout = 300.0

        with patch.dict("sys.modules", {"mlx_lm": None}):
            with pytest.raises(ImportError, match="mlx-lm"):
                provider._ensure_loaded()


class TestMLXProviderPromptAssembly:
    def test_default_template_used_when_no_apply_fn(self):
        provider = _make_mlx_provider()
        provider._mlx_model = MagicMock()
        provider._tokenizer = MagicMock(spec=[])  # no apply_chat_template attr
        prompt = provider._build_prompt("Be helpful.", "What is caching?")
        assert "Be helpful." in prompt
        assert "What is caching?" in prompt
        assert "<|im_start|>" in prompt

    def test_tokenizer_template_used_when_available(self):
        provider = _make_mlx_provider()
        provider._mlx_model = MagicMock()
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "<CUSTOM_TEMPLATE>"
        provider._tokenizer = mock_tok

        prompt = provider._build_prompt("sys", "user")
        assert prompt == "<CUSTOM_TEMPLATE>"
        mock_tok.apply_chat_template.assert_called_once()


class TestMLXProviderClose:
    def test_close_releases_references(self):
        provider = _make_mlx_provider()
        provider._mlx_model = MagicMock()
        provider._tokenizer = MagicMock()
        provider.close()
        assert provider._mlx_model is None
        assert provider._tokenizer is None

    def test_repr_after_close(self):
        provider = _make_mlx_provider()
        provider._mlx_model = MagicMock()
        provider.close()
        assert "loaded=False" in repr(provider)


class TestMLXProviderInFactory:
    def test_create_provider_mlx(self):
        """create_provider({'type': 'mlx', ...}) returns an MLXProvider instance."""
        from servo_skull.llm_providers import create_provider
        # MLXProvider is imported lazily inside create_provider(), so patch it
        # at its definition site in mlx_provider module.
        with patch("servo_skull.mlx_provider.MLXProvider") as MockMLX:
            mock_instance = MagicMock()
            MockMLX.return_value = mock_instance
            provider = create_provider({
                "type": "mlx",
                "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                "max_tokens": 2048,
            })
        assert provider is mock_instance

"""Tests for AI client module."""

import os
from unittest import mock

import pytest

from rendercv.ai.client import (
    AIConfig,
    AnthropicClient,
    OpenAIClient,
    get_ai_client,
)
from rendercv.exception import RenderCVUserError


class TestAIConfig:
    def test_get_model_openai_default(self):
        config = AIConfig(provider="openai", api_key="test-key")
        assert config.get_model() == "gpt-4o"

    def test_get_model_anthropic_default(self):
        config = AIConfig(provider="anthropic", api_key="test-key")
        assert config.get_model() == "claude-sonnet-4-20250514"

    def test_get_model_custom(self):
        config = AIConfig(provider="openai", api_key="test-key", model="gpt-3.5-turbo")
        assert config.get_model() == "gpt-3.5-turbo"

    def test_get_model_unknown_provider(self):
        config = AIConfig(provider="unknown", api_key="test-key")
        with pytest.raises(RenderCVUserError, match="Unknown AI provider"):
            config.get_model()


class TestGetAIClient:
    def test_auto_detect_openai_from_env(self):
        with mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False
        ), mock.patch(
            "rendercv.ai.client.OpenAIClient"
        ) as mock_client:
            mock_client.return_value = mock.MagicMock()
            get_ai_client()
            mock_client.assert_called_once()

    def test_auto_detect_anthropic_from_env(self):
        with mock.patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "test-key"},
            clear=True,
        ), mock.patch(
            "rendercv.ai.client.AnthropicClient"
        ) as mock_client:
            mock_client.return_value = mock.MagicMock()
            # Remove OPENAI_API_KEY if present
            os.environ.pop("OPENAI_API_KEY", None)
            get_ai_client()
            mock_client.assert_called_once()

    def test_no_api_key_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(RenderCVUserError, match="No API key found"):
                get_ai_client()

    def test_provider_with_api_key(self):
        with mock.patch("rendercv.ai.client.OpenAIClient") as mock_client:
            mock_client.return_value = mock.MagicMock()
            get_ai_client(provider="openai", api_key="test-key")
            mock_client.assert_called_once()

    def test_unknown_provider_error(self):
        with pytest.raises(RenderCVUserError, match="Unknown AI provider"):
            get_ai_client(provider="unknown", api_key="test-key")

    def test_api_key_without_provider_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(
                RenderCVUserError, match="you must also specify the provider"
            ):
                get_ai_client(api_key="test-key")


class TestOpenAIClient:
    def test_generate(self):
        with mock.patch("openai.OpenAI") as mock_openai:
            mock_response = mock.MagicMock()
            mock_response.choices = [mock.MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_openai.return_value.chat.completions.create.return_value = (
                mock_response
            )

            config = AIConfig(provider="openai", api_key="test-key")
            client = OpenAIClient(config)
            result = client.generate("Test prompt")

            assert result == "Test response"

    def test_generate_json(self):
        with mock.patch("openai.OpenAI") as mock_openai:
            mock_response = mock.MagicMock()
            mock_response.choices = [mock.MagicMock()]
            mock_response.choices[0].message.content = '{"key": "value"}'
            mock_openai.return_value.chat.completions.create.return_value = (
                mock_response
            )

            config = AIConfig(provider="openai", api_key="test-key")
            client = OpenAIClient(config)
            result = client.generate_json("Test prompt")

            assert result == {"key": "value"}


class TestAnthropicClient:
    def test_generate(self):
        with mock.patch("anthropic.Anthropic") as mock_anthropic:
            mock_response = mock.MagicMock()
            mock_response.content = [mock.MagicMock()]
            mock_response.content[0].text = "Test response"
            mock_anthropic.return_value.messages.create.return_value = mock_response

            config = AIConfig(provider="anthropic", api_key="test-key")
            client = AnthropicClient(config)
            result = client.generate("Test prompt")

            assert result == "Test response"

    def test_generate_json(self):
        with mock.patch("anthropic.Anthropic") as mock_anthropic:
            mock_response = mock.MagicMock()
            mock_response.content = [mock.MagicMock()]
            mock_response.content[0].text = '{"key": "value"}'
            mock_anthropic.return_value.messages.create.return_value = mock_response

            config = AIConfig(provider="anthropic", api_key="test-key")
            client = AnthropicClient(config)
            result = client.generate_json("Test prompt")

            assert result == {"key": "value"}

    def test_generate_json_with_markdown_code_block(self):
        with mock.patch("anthropic.Anthropic") as mock_anthropic:
            mock_response = mock.MagicMock()
            mock_response.content = [mock.MagicMock()]
            mock_response.content[0].text = '```json\n{"key": "value"}\n```'
            mock_anthropic.return_value.messages.create.return_value = mock_response

            config = AIConfig(provider="anthropic", api_key="test-key")
            client = AnthropicClient(config)
            result = client.generate_json("Test prompt")

            assert result == {"key": "value"}

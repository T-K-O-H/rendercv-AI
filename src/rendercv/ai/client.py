"""AI client abstraction for LLM providers (OpenAI, Anthropic)."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rendercv.exception import RenderCVUserError

if TYPE_CHECKING:
    from typing import Any


@dataclass
class AIConfig:
    """Configuration for AI client."""

    provider: str  # "openai" or "anthropic"
    api_key: str
    model: str | None = None

    def get_model(self) -> str:
        """Get the model to use, defaulting based on provider."""
        if self.model:
            return self.model
        if self.provider == "openai":
            return "gpt-4o"
        if self.provider == "anthropic":
            return "claude-sonnet-4-20250514"
        message = f"Unknown AI provider: {self.provider}"
        raise RenderCVUserError(message)


class AIClient(ABC):
    """Abstract base class for AI clients."""

    def __init__(self, config: AIConfig):
        self.config = config

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate text using the AI model.

        Args:
            prompt: The user prompt to send to the model.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature (0-1).
            max_tokens: Maximum tokens in response.

        Returns:
            The generated text response.
        """

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        """Generate JSON using the AI model.

        Args:
            prompt: The user prompt to send to the model.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature (0-1).
            max_tokens: Maximum tokens in response.

        Returns:
            The parsed JSON response as a dictionary.
        """


class OpenAIClient(AIClient):
    """OpenAI API client implementation."""

    def __init__(self, config: AIConfig):
        super().__init__(config)
        try:
            import openai  # noqa: PLC0415
        except ImportError as e:
            message = (
                "OpenAI package is not installed. Install it with: pip install openai"
            )
            raise RenderCVUserError(message) from e
        self.client = openai.OpenAI(api_key=config.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.get_model(),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.get_model(),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)


class AnthropicClient(AIClient):
    """Anthropic API client implementation."""

    def __init__(self, config: AIConfig):
        super().__init__(config)
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as e:
            message = (
                "Anthropic package is not installed. Install it with: pip install"
                " anthropic"
            )
            raise RenderCVUserError(message) from e
        self.client = anthropic.Anthropic(api_key=config.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.config.get_model(),
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        # Anthropic doesn't support temperature=0, use a small value
        if temperature > 0:
            kwargs["temperature"] = temperature

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        # Enhance prompt to request JSON output
        json_prompt = (
            f"{prompt}\n\nRespond with valid JSON only."
            " No explanation or markdown formatting."
        )

        kwargs: dict[str, Any] = {
            "model": self.config.get_model(),
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": json_prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature > 0:
            kwargs["temperature"] = temperature

        response = self.client.messages.create(**kwargs)
        content = response.content[0].text

        # Extract JSON from response (handle potential markdown code blocks)
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        return json.loads(content)


def get_ai_client(
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> AIClient:
    """Factory function to create an AI client.

    Args:
        provider: The AI provider ("openai" or "anthropic"). If None, tries to
            auto-detect from available API keys.
        api_key: The API key. If None, reads from environment variables.
        model: The model to use. If None, uses provider defaults.

    Returns:
        An AIClient instance for the specified provider.

    Raises:
        RenderCVUserError: If provider is unknown or API key is missing.
    """
    # Auto-detect provider from environment if not specified
    if provider is None:
        if api_key:
            raise RenderCVUserError(
                "When providing an API key, you must also specify the provider"
                " (--provider openai or --provider anthropic)"
            )
        # Check environment variables
        if os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
            api_key = os.environ["OPENAI_API_KEY"]
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
            api_key = os.environ["ANTHROPIC_API_KEY"]
        else:
            raise RenderCVUserError(
                "No API key found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY"
                " environment variable, or use --api-key with --provider option."
            )

    # Get API key from environment if not provided
    if api_key is None:
        env_var = f"{provider.upper()}_API_KEY"
        api_key = os.environ.get(env_var)
        if not api_key:
            message = (
                f"API key not found. Please set {env_var} environment variable"
                " or use --api-key option."
            )
            raise RenderCVUserError(message)

    config = AIConfig(provider=provider, api_key=api_key, model=model)

    if provider == "openai":
        return OpenAIClient(config)
    if provider == "anthropic":
        return AnthropicClient(config)
    message = f"Unknown AI provider: {provider}. Supported providers: openai, anthropic"
    raise RenderCVUserError(message)

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain.chat_models import init_chat_model

_DEFAULT_MODEL = "gemini-2.5-pro"
_DEFAULT_PROVIDER = "google_genai"


@dataclass(frozen=True)
class ModelSpec:
    """Normalised representation of the chat model + provider."""

    provider: str
    model: str

    @property
    def identifier(self) -> str:
        """Return a provider-prefixed identifier (provider:model)."""

        return f"{self.provider}:{self.model}" if self.provider else self.model


def _default_model() -> str:
    return (
        os.environ.get("EMAIL_ASSISTANT_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or _DEFAULT_MODEL
    )


def _default_provider() -> str:
    return os.environ.get("EMAIL_ASSISTANT_MODEL_PROVIDER", _DEFAULT_PROVIDER)


def normalize_model_spec(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    default_model: str | None = None,
    default_provider: str | None = None,
) -> ModelSpec:
    """Return a normalised provider/model pair for Gemini chat models.

    Accepts model strings with optional provider prefixes (``provider:model``)
    or Vertex-style ``models/<id>`` paths and ensures the provider defaults to
    Google GenAI unless explicitly overridden.
    """

    effective_model = (model or "").strip() or default_model or _default_model()
    provider = (
        model_provider
        or (default_provider or _default_provider())
        or ""
    )

    if ":" in effective_model:
        candidate_provider, candidate_model = effective_model.split(":", 1)
        candidate_provider = candidate_provider.strip()
        candidate_model = candidate_model.strip()
        if candidate_provider:
            provider = candidate_provider
        effective_model = candidate_model or default_model or _default_model()

    if effective_model.startswith("models/"):
        remainder = effective_model.split("/", 1)[1]
        if remainder:
            effective_model = remainder
        else:
            effective_model = default_model or _default_model()

    if not effective_model:
        effective_model = default_model or _default_model()

    return ModelSpec(provider=provider, model=effective_model)


def format_model_identifier(model: str | None = None, *, provider: str | None = None) -> str:
    """Return a provider-prefixed identifier string for logging."""

    spec = normalize_model_spec(model, model_provider=provider)
    return spec.identifier


def model_spec(model: str | None = None, *, provider: str | None = None) -> ModelSpec:
    """Convenience helper that exposes the normalised ``ModelSpec``."""

    return normalize_model_spec(model, model_provider=provider)


def get_llm(temperature: float = 0.0, **kwargs):
    """Return a configured Gemini chat model instance.

    Args:
        temperature: The temperature to use for the LLM.
        **kwargs: Additional parameters passed through to ``init_chat_model``.

    Returns:
        A configured ``BaseChatModel`` instance from LangChain's factory.
    """

    # Allow explicit model override via kwargs, fallback to env var (2.5 series by default)
    raw_model = kwargs.pop("model", None)
    provider_override = kwargs.pop("model_provider", None)
    spec = normalize_model_spec(raw_model, model_provider=provider_override)

    # Gemini now natively supports system messages; disable the legacy conversion layer to
    # avoid the noisy "Convert_system_message_to_human will be deprecated" warning.
    kwargs.setdefault("convert_system_message_to_human", False)

    return init_chat_model(
        spec.model,
        model_provider=spec.provider,
        temperature=temperature,
        **kwargs,
    )

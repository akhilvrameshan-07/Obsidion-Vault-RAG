"""
llm_provider.py — Provider-agnostic LLM abstraction for the RAG assistant.

Swap models by changing environment variables only — no code changes:

    LLM_PROVIDER=gemini|openai|anthropic|huggingface|ollama
    LLM_MODEL=<model-name>
    LLM_API_KEY=<key>          # not needed for huggingface (local) / ollama (local)
    LLM_BASE_URL=<url>         # ollama endpoint, or any OpenAI-compatible endpoint (vLLM, LM Studio, etc.)
    LLM_TEMPERATURE=0
    LLM_MAX_TOKENS=1024

Design notes
------------
LangChain's chat model classes already share one interface:
    response = model.invoke(messages)   # -> AIMessage with .content
So this module doesn't reinvent an interface — it does the two things that
were actually missing from the original code:

1. A factory that picks the right LangChain class for a given provider name,
   importing that provider's SDK lazily (so you don't need openai+anthropic+
   transformers installed just to use Gemini).
2. Provider-agnostic error handling. The original generate_answer() had
   Gemini-specific substring checks ("API_KEY_INVALID", etc.) baked in,
   which silently stopped being useful the moment you swapped providers.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("gemini", "openai", "anthropic", "huggingface", "ollama")

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
    "huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",
    "ollama": "llama3",
}


class LLMProviderError(RuntimeError):
    """Raised when a chat model can't be constructed (bad config, missing key/package, etc.)."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024

    @classmethod
    def load_from_env(cls) -> "LLMConfig":
        load_dotenv()

        provider = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise LLMProviderError(
                f"Unknown LLM_PROVIDER '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}."
            )

        # Generic key first; fall back to provider-specific names so existing
        # .env files (e.g. from before this module existed) keep working.
        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
        )

        model = (
            os.getenv("LLM_MODEL")
            or os.getenv("GEMINI_MODEL")
            or DEFAULT_MODELS.get(provider, "")
        )

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        )


def get_chat_model(config: LLMConfig) -> BaseChatModel:
    """
    Factory: returns a ready-to-use LangChain chat model for config.provider.
    Build this ONCE at startup and reuse it across requests — don't
    reconstruct it per-call.
    """
    provider = config.provider

    if provider == "gemini":
        if not config.api_key:
            raise LLMProviderError("gemini selected but no API key found (LLM_API_KEY / GEMINI_API_KEY).")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            api_key=config.api_key,
        )

    if provider == "openai":
        if not config.api_key:
            raise LLMProviderError("openai selected but no API key found (LLM_API_KEY / OPENAI_API_KEY).")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
            base_url=config.base_url,  # doubles as an OpenAI-compatible slot for vLLM / LM Studio / etc.
        )

    if provider == "anthropic":
        if not config.api_key:
            raise LLMProviderError("anthropic selected but no API key found (LLM_API_KEY / ANTHROPIC_API_KEY).")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
        )

    if provider == "huggingface":
        # Runs locally via transformers — no API key needed, but needs the
        # model weights downloaded/cached and enough local compute.
        from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
        llm = HuggingFacePipeline.from_model_id(
            model_id=config.model,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature or 0.01,  # 0.0 is invalid for many HF pipelines
            },
        )
        return ChatHuggingFace(llm=llm)

    if provider == "ollama":
        # Local self-hosted models (llama3, mistral, qwen, ...) via an Ollama server.
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.model,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )

    raise LLMProviderError(f"Unhandled provider '{provider}'.")  # unreachable given the check above


def _normalize_content(content: Any) -> str:
    """Some providers return content as a list of parts instead of a plain
    string (e.g. [{"type": "text", "text": "..."}]). Flatten to one string."""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return content or ""


def generate_answer(
    chat_model: BaseChatModel,
    prompt_messages: List[BaseMessage],
    provider: str,
    model_name: str,
) -> str:
    """
    Invoke any LangChain chat model and return clean text, with
    provider-agnostic, user-friendly error mapping (instead of one
    hardcoded to a single provider's error strings).
    """
    try:
        response = chat_model.invoke(prompt_messages)
        content = _normalize_content(response.content)

        if not content.strip():
            logger.error(f"Empty/unrecognized response shape from {provider}: {response.content!r}")
            return f"API Error: Received an empty or unrecognized response from {provider}."

        return content.strip()

    except Exception as e:
        error_msg = str(e)
        logger.error(f"{provider} ({model_name}) call failed: {error_msg}")

        lowered = error_msg.lower()
        if any(t in lowered for t in ("api_key_invalid", "401", "unauthorized", "invalid api key")):
            return f"API Error: The {provider} API key provided is invalid or unauthorized."
        if any(t in lowered for t in ("quota", "429", "rate limit")):
            return f"API Error: Rate limit exceeded or quota exhausted for {provider}. Please try again later."
        if any(t in lowered for t in ("network", "connection", "timeout")):
            return "API Error: Network connection failure. Please check your internet connectivity."
        if any(t in lowered for t in ("not found", "404")):
            return f"API Error: Model '{model_name}' not found or not available for {provider}."
        return f"API Error: Failed to retrieve answer from {provider} ({error_msg})."
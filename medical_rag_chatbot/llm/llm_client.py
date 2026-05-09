"""
llm_client.py
-------------
Unified LLM client supporting:
  - OpenAI  (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
  - Google  (gemini-1.5-pro, gemini-1.5-flash)

Usage:
    from llm.llm_client import get_llm_client
    client = get_llm_client()
    response = client.chat(messages, system_prompt)
    for token in client.stream(messages, system_prompt):
        print(token, end="", flush=True)
"""

import os
from abc import ABC, abstractmethod
from typing import Generator

from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────
class LLMClient(ABC):

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        system_prompt: str = "",
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        """Blocking completion. Returns full response string."""

    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        system_prompt: str = "",
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """Streaming completion. Yields text tokens."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI client
# ─────────────────────────────────────────────────────────────────────────────
class OpenAIClient(LLMClient):

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")

        self._client = OpenAI(api_key=api_key)
        self._model  = model

    @property
    def model_name(self) -> str:
        return self._model

    def _build_oai_messages(
        self, messages: list[dict], system_prompt: str
    ) -> list[dict]:
        oai_msgs = []
        if system_prompt:
            oai_msgs.append({"role": "system", "content": system_prompt})
        for m in messages:
            oai_msgs.append({"role": m["role"], "content": m["content"]})
        return oai_msgs

    def chat(
        self,
        messages: list[dict],
        system_prompt: str = "",
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._build_oai_messages(messages, system_prompt),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def stream(
        self,
        messages: list[dict],
        system_prompt: str = "",
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=self._build_oai_messages(messages, system_prompt),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ─────────────────────────────────────────────────────────────────────────────
# Google Gemini client
# ─────────────────────────────────────────────────────────────────────────────
class GeminiClient(LLMClient):

    def __init__(self, model: str = "gemini-1.5-flash"):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment.")

        genai.configure(api_key=api_key)
        self._genai  = genai
        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name

    def _to_gemini_history(
        self, messages: list[dict]
    ) -> tuple[list[dict], str]:
        """Split messages into history + last user message."""
        history = []
        for m in messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            history.append({"role": role, "parts": [m["content"]]})
        last_msg = messages[-1]["content"] if messages else ""
        return history, last_msg

    def chat(
        self,
        messages: list[dict],
        system_prompt: str = "",
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        model = self._genai.GenerativeModel(
            self._model_name,
            system_instruction=system_prompt or None,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        history, last_msg = self._to_gemini_history(messages)
        chat = model.start_chat(history=history)
        response = chat.send_message(last_msg)
        return response.text

    def stream(
        self,
        messages: list[dict],
        system_prompt: str = "",
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        model = self._genai.GenerativeModel(
            self._model_name,
            system_instruction=system_prompt or None,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        history, last_msg = self._to_gemini_history(messages)
        chat = model.start_chat(history=history)
        response = chat.send_message(last_msg, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
class GroqClient(LLMClient):

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Install groq: pip install groq")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment.")

        self._client = Groq(api_key=api_key)
        self._model  = model

    @property
    def model_name(self) -> str:
        return self._model

    def chat(self, messages, system_prompt="", max_tokens=800, temperature=0.2) -> str:
        oai_msgs = []
        if system_prompt:
            oai_msgs.append({"role": "system", "content": system_prompt})
        for m in messages:
            oai_msgs.append({"role": m["role"], "content": m["content"]})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=oai_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def stream(self, messages, system_prompt="", max_tokens=800, temperature=0.2):
        oai_msgs = []
        if system_prompt:
            oai_msgs.append({"role": "system", "content": system_prompt})
        for m in messages:
            oai_msgs.append({"role": m["role"], "content": m["content"]})

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=oai_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

# ─────────────────────────────────────────────────────────────────────────────
# Mock client for testing without API keys
# ─────────────────────────────────────────────────────────────────────────────
class MockLLMClient(LLMClient):
    """Returns canned responses — useful for UI testing."""

    @property
    def model_name(self) -> str:
        return "mock-model"

    def chat(self, messages, system_prompt="", max_tokens=800, temperature=0.2) -> str:
        query = messages[-1]["content"] if messages else ""
        return (
            f"**[DEMO MODE — No API key configured]**\n\n"
            f"This is a mock response for: *{query[:80]}*\n\n"
            "To get real answers, add your `OPENAI_API_KEY` or `GOOGLE_API_KEY` "
            "to the `.env` file and restart the app."
        )

    def stream(self, messages, system_prompt="", max_tokens=800, temperature=0.2):
        yield self.chat(messages, system_prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
_client_cache: dict[str, LLMClient] = {}


def get_llm_client(model: str | None = None) -> LLMClient:
    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    if model in _client_cache:
        return _client_cache[model]

    try:
        if model.startswith("gpt"):
            client = OpenAIClient(model=model)
        elif model.startswith("gemini"):
            client = GeminiClient(model=model)
        elif model in ("groq", "llama-3.1-8b-instant", "llama-3.3-70b-versatile" "mixtral-8x7b-32768"):
            actual_model = "llama-3.1-8b-instant" if model == "groq" else model
            client = GroqClient(model=actual_model)
        elif model == "mock":
            client = MockLLMClient()
        else:
            raise ValueError(f"Unknown model: {model}")
    except (ValueError, ImportError) as e:
        print(f"[LLM] Warning: {e}. Falling back to MockLLMClient.")
        client = MockLLMClient()

    _client_cache[model] = client
    return client
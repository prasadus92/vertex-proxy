"""Translation between the OpenAI Chat Completions API and the Anthropic Messages API."""

from .request import openai_to_anthropic
from .response import anthropic_to_openai
from .stream import StreamTranslator

__all__ = ["StreamTranslator", "anthropic_to_openai", "openai_to_anthropic"]

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING, Iterable, cast

from ..amy.core.models import Message

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


@dataclass
class OpenAIResponder:
    api_key: str
    model: str
    max_output_tokens: int = 300
    temperature: float = 0.2

    def generate_reply(self, messages: list[Message], cancel_event: threading.Event) -> str:
        if cancel_event.is_set():
            return ""

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        payload = cast(
            "list[ChatCompletionMessageParam]",
            [{"role": message.role, "content": message.content} for message in messages],
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=payload,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )

        if cancel_event.is_set():
            return ""

        choice = response.choices[0]
        content = choice.message.content or ""
        return content.strip()

    @staticmethod
    def count_prompt_tokens(messages: Iterable[Message]) -> int:
        return sum(max(1, len(message.content.split())) for message in messages)

    @staticmethod
    def estimate_cost_usd(token_count: int, input_rate: float = 0.00015) -> float:
        return round(token_count * input_rate, 6)

__all__ = ["OpenAIResponder"]

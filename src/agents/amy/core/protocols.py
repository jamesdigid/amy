from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .models import Message

if TYPE_CHECKING:
    import threading

    from ..memory import MemoryDecision, MemoryDraft


class Responder(Protocol):
    def generate_reply(self, messages: list[Message], cancel_event: threading.Event) -> str: ...


class Speaker(Protocol):
    def speak(self, text: str) -> None: ...

    def stop(self) -> None: ...


class MemoryStoreProtocol(Protocol):
    def retrieve_context(self, prompt: str, limit: int = 3) -> str: ...

    def draft_from_prompt(self, prompt: str, subject: str | None = None) -> MemoryDraft | None: ...

    def save_draft(self, draft: MemoryDraft) -> Path: ...


class MemoryClassifierProtocol(Protocol):
    def classify(self, prompt: str, cancel_event: threading.Event) -> MemoryDecision: ...

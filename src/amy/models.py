from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

Role = Literal["system", "user", "assistant"]


class AssistantPhase(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    THINKING = "thinking"
    SPEAKING = "speaking"
    COOLDOWN = "cooldown"
    AWAITING_USER_RESPONSE = "awaiting_user_response"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class ConversationTurn:
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AssistantStatus:
    phase: AssistantPhase = AssistantPhase.IDLE
    active_conversation: bool = False
    paused: bool = False
    wake_word: str = "amy"
    last_user_text: str = ""
    last_assistant_text: str = ""
    error_message: str = ""


@dataclass
class AssistantResponse:
    text: str
    was_cancelled: bool = False

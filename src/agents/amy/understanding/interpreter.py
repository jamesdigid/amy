
from __future__ import annotations

from dataclasses import dataclass, field
import re

from ..models import Message

PAUSE_COMMANDS = frozenset({"amy pause", "pause conversation", "stop listening", "pause"})
RESUME_COMMANDS = frozenset({"amy resume", "resume conversation", "resume"})
CUT_COMMANDS = frozenset({"amy cut", "cut channel", "cut", "stop"})
STATUS_PHRASES = (
    "status check",
    "check your status",
    "check status",
    "what is your status",
    "whats your status",
    "what's your status",
    "how are you doing",
)
SEARCH_PREFIXES = (
    "search web for ",
    "search the web for ",
    "search for ",
    "web for ",
    "look up ",
    "look up the web for ",
    "find information about ",
    "find out about ",
    "find web results for ",
    "web search for ",
)
FOLLOW_UP_SIGNALS = (
    "can you",
    "could you",
    "would you",
    "do you",
    "did you",
    "should you",
    "shall we",
    "what would you",
    "what do you",
    "what can you",
    "would you like",
    "do you want",
    "let me know",
    "tell me",
    "want me to",
)
WRAP_UP_SIGNALS = (
    "anything else i can help you with",
    "anything else i can do",
    "anything else i can help",
    "let me know if you need anything else",
    "let me know if i can help with anything else",
    "is there anything else",
)
MEMORY_REQUEST_SIGNALS = (
    "remember this",
    "remember that",
    "remember for later",
    "save this for later",
    "don't forget",
    "dont forget",
    "note this",
    "keep this in mind",
)
MEMORY_SIGNALS = (
    "i am ",
    "i'm ",
    "my ",
    "we are ",
    "we're ",
    "our ",
    "prefer ",
    "favorite ",
    "favourite ",
    "birthday ",
    "anniversary ",
)
QUESTION_STARTERS = (
    "who ",
    "what ",
    "when ",
    "where ",
    "why ",
    "how ",
    "is ",
    "are ",
    "can ",
    "could ",
    "should ",
    "would ",
    "do ",
    "does ",
    "did ",
)
SHORT_INTERRUPT_WORDS = ("pause", "resume", "cut", "stop")


@dataclass(frozen=True)
class TranscriptInterpreter:
    wake_word: str
    acknowledgement_echoes: frozenset[str] = field(
        default_factory=lambda: frozenset({"yes", "yeah", "yep", "no", "ok", "okay", "sure", "right"}),
        init=False,
        repr=False,
    )

    def normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def strip_wake_word(self, text: str) -> str:
        pattern = re.compile(rf"^\s*{re.escape(self.wake_word)}[\s,.:;-]*", re.IGNORECASE)
        return pattern.sub("", text).strip()

    def starts_with_wake_word(self, text: str) -> bool:
        return re.match(rf"^{re.escape(self.wake_word)}\b", text) is not None

    def strip_acknowledgement_prefix(self, text: str) -> str:
        pattern = re.compile(r"^\s*(?:amy\s+)?here[\s,.:;-]*", re.IGNORECASE)
        return pattern.sub("", text).strip()

    def normalize_echo_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

    def is_pause_command(self, text: str) -> bool:
        return self.normalize_text(text) in PAUSE_COMMANDS

    def is_resume_command(self, text: str) -> bool:
        return self.normalize_text(text) in RESUME_COMMANDS

    def is_cut_command(self, text: str) -> bool:
        return self.normalize_text(text) in CUT_COMMANDS

    def is_status_command(self, text: str) -> bool:
        normalized = self.normalize_text(text)
        return any(phrase in normalized for phrase in STATUS_PHRASES)

    def looks_like_short_interrupt(self, normalized: str) -> bool:
        words = normalized.split()
        if not words or len(words) > 4:
            return False
        return any(re.search(rf"\b{word}\b", normalized) for word in SHORT_INTERRUPT_WORDS)

    def is_acknowledgement_echo(
        self,
        normalized: str,
        *,
        active_conversation: bool,
        last_assistant_text: str,
    ) -> bool:
        if normalized == "amy here":
            return True
        if not active_conversation:
            return False
        normalized_echo = self.normalize_echo_text(normalized)
        if not last_assistant_text:
            return False
        normalized_assistant = self.normalize_echo_text(last_assistant_text)
        return normalized_echo in self.acknowledgement_echoes and normalized_echo == normalized_assistant

    def estimate_tokens(self, messages: list[Message]) -> int:
        return sum(max(1, len(message.content.split())) for message in messages)

    def extract_search_query(self, prompt: str) -> str:
        normalized = self.normalize_text(prompt)
        for prefix in SEARCH_PREFIXES:
            if normalized.startswith(prefix):
                return prompt[len(prefix) :].strip(" ,.:;-")

        if self.should_web_search(normalized):
            return prompt.strip()
        return ""

    def should_web_search(self, normalized_prompt: str) -> bool:
        search_signals = (
            "latest",
            "recent",
            "current",
            "today",
            "news",
            "weather",
            "stocks",
            "stock",
            "price",
            "prices",
        )
        return any(signal in normalized_prompt for signal in search_signals)

    def reply_expects_follow_up(self, reply: str) -> bool:
        normalized = self.normalize_echo_text(reply)
        if "?" in reply:
            return True
        return any(signal in normalized for signal in FOLLOW_UP_SIGNALS)

    def reply_ends_session_immediately(self, reply: str) -> bool:
        normalized = self.normalize_echo_text(reply)
        return any(signal in normalized for signal in WRAP_UP_SIGNALS)

    def limit_follow_up_questions(self, reply: str) -> str:
        first_question_mark = reply.find("?")
        if first_question_mark < 0:
            return reply.strip()

        if reply.count("?") <= 1:
            return reply.strip()

        return reply[: first_question_mark + 1].strip()

    def is_memory_request(self, text: str) -> bool:
        return any(signal in text for signal in MEMORY_REQUEST_SIGNALS)

    def looks_like_direct_question(self, normalized: str) -> bool:
        if normalized.endswith("?"):
            return True
        return normalized.startswith(QUESTION_STARTERS)

    def should_consider_memory(self, prompt: str) -> bool:
        normalized = self.normalize_text(prompt)
        if self.is_memory_request(normalized):
            return True
        if self.looks_like_direct_question(normalized):
            return False
        if self.extract_search_query(prompt):
            return False
        return any(signal in normalized for signal in MEMORY_SIGNALS)

__all__ = ["TranscriptInterpreter"]

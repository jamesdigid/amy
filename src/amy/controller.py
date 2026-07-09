from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
import threading
from typing import Callable, Protocol

from .context import PromptBuilder
from .models import AssistantPhase, AssistantStatus, ConversationTurn, Message
from .web_search import SearchResult, WebSearcher


class Responder(Protocol):
    def generate_reply(self, messages: list[Message], cancel_event: threading.Event) -> str: ...


class Speaker(Protocol):
    def speak(self, text: str) -> None: ...

    def stop(self) -> None: ...


@dataclass
class AssistantController:
    prompt_builder: PromptBuilder
    responder: Responder
    speaker: Speaker
    wake_word: str
    web_search: WebSearcher | None = None
    web_search_limit: int = 4
    idle_timeout_seconds: float = 10.0
    usage_logger: Callable[[int, float], None] | None = None
    acknowledgment_callback: Callable[[], None] | None = None
    acknowledgment_stop_callback: Callable[[], None] | None = None
    status: AssistantStatus = field(default_factory=AssistantStatus)
    turns: list[ConversationTurn] = field(default_factory=lambda: list[ConversationTurn]())
    _cancel_event: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _idle_timer: threading.Timer | None = field(default=None, init=False)
    _logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__), init=False, repr=False)

    def pause(self) -> None:
        with self._lock:
            self._logger.debug("pause requested")
            self._cancel_idle_timer_locked()
            self._cancel_event.set()
            self.speaker.stop()
            self.status.active_conversation = False
            if self.status.active_conversation:
                self.status.phase = AssistantPhase.LISTENING
            else:
                self.status.phase = AssistantPhase.IDLE
            self.status.paused = False

    def resume(self) -> None:
        with self._lock:
            self._logger.debug("resume requested")
            self.status.paused = False
            self.status.phase = AssistantPhase.LISTENING

    def cut_channel(self) -> None:
        with self._lock:
            self._logger.debug("cut requested")
            self._cancel_idle_timer_locked()
            self._cancel_event.set()
            self.speaker.stop()
            self.status.active_conversation = False
            self.status.paused = True
            self.status.phase = AssistantPhase.PAUSED

    def stop(self) -> None:
        with self._lock:
            self._cancel_idle_timer_locked()
            self._cancel_event.set()
            self.speaker.stop()
            self.status.active_conversation = False
            self.status.phase = AssistantPhase.IDLE
            self.status.paused = False

    def process_transcript(self, transcript: str) -> str | None:
        normalized = self._normalize_text(transcript)
        if not normalized:
            return None

        self._logger.debug("received transcript: raw=%r normalized=%r", transcript, normalized)

        if self._is_acknowledgement_echo(normalized):
            self._logger.debug("dropping acknowledgement echo")
            return None

        if self._is_pause_command(normalized):
            self._logger.debug("pause command matched")
            self.pause()
            return None
        if self._is_resume_command(normalized):
            self._logger.debug("resume command matched")
            self.resume()
            return None
        if self._is_cut_command(normalized):
            self._logger.debug("cut command matched")
            self.cut_channel()
            return None

        with self._lock:
            if self.status.paused:
                self._logger.debug("dropping transcript because assistant is paused")
                return None

            if self.status.active_conversation:
                self._cancel_idle_timer_locked()

            prompt = transcript.strip()
            prompt = self._strip_acknowledgement_prefix(prompt)
            if not self.status.active_conversation:
                if not self._starts_with_wake_word(normalized):
                    self._logger.debug("dropping transcript because wake word did not match")
                    return None
                prompt = self._strip_wake_word(prompt)
                prompt = self._strip_acknowledgement_prefix(prompt)
                if not prompt:
                    self._logger.debug("wake word alone; acknowledging only")
                    self.status.active_conversation = True
                    self.status.phase = AssistantPhase.LISTENING
                    self._stop_acknowledgement_loop()
                    self.speaker.speak("Amy here")
                    self._schedule_idle_timeout_locked()
                    return None
                self.status.active_conversation = True
                self.status.phase = AssistantPhase.RECORDING
                self._logger.debug("wake word matched; capturing prompt")

            web_context = ""
            search_query = self._extract_search_query(prompt)
            if self.web_search is not None and search_query:
                self._logger.debug("web search triggered: %r", search_query)
                self._emit_acknowledgement()
                web_results = self.web_search.search(search_query, self.web_search_limit)
                web_context = self._format_web_context(search_query, web_results)

            self.status.phase = AssistantPhase.THINKING
            self.status.last_user_text = prompt
            self.turns.append(ConversationTurn(role="user", content=prompt))
            self._cancel_event.clear()

        messages = self.prompt_builder.build_messages(self.turns[:-1], prompt, web_context=web_context)
        if self.usage_logger is not None:
            token_count = self._estimate_tokens(messages)
            self.usage_logger(token_count, token_count * 0.00015)
        self._logger.debug("sending %d messages to responder", len(messages))
        reply = self.responder.generate_reply(messages, self._cancel_event)

        if self._cancel_event.is_set():
            self._logger.debug("reply cancelled before speech")
            return None

        with self._lock:
            self.status.phase = AssistantPhase.SPEAKING
            self.status.last_assistant_text = reply
            self.turns.append(ConversationTurn(role="assistant", content=reply))

        self._logger.debug("speaking reply: %r", reply)
        self._stop_acknowledgement_loop()
        self.speaker.speak(reply)

        with self._lock:
            if not self._cancel_event.is_set():
                self._logger.debug("reply complete; waiting for follow-up")
                self.status.active_conversation = True
                self.status.phase = AssistantPhase.LISTENING
                self._schedule_idle_timeout_locked()
        return reply

    def get_status(self) -> AssistantStatus:
        return self.status

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _strip_wake_word(self, text: str) -> str:
        pattern = re.compile(rf"^\s*{re.escape(self.wake_word)}[\s,.:;-]*", re.IGNORECASE)
        return pattern.sub("", text).strip()

    def _starts_with_wake_word(self, text: str) -> bool:
        return re.match(rf"^{re.escape(self.wake_word)}\b", text) is not None

    def _is_pause_command(self, text: str) -> bool:
        return text in {"amy pause", "pause conversation", "stop listening", "pause"}

    def _is_resume_command(self, text: str) -> bool:
        return text in {"amy resume", "resume conversation", "resume"}

    def _is_cut_command(self, text: str) -> bool:
        return text in {"amy cut", "cut channel", "cut", "stop"}

    def is_interrupt_command(self, transcript: str) -> bool:
        normalized = self._normalize_text(transcript)
        return self._looks_like_short_interrupt(normalized)

    def _estimate_tokens(self, messages: list[Message]) -> int:
        return sum(max(1, len(message.content.split())) for message in messages)

    def _extract_search_query(self, prompt: str) -> str:
        normalized = self._normalize_text(prompt)
        prefixes = (
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
        for prefix in prefixes:
            if normalized.startswith(prefix):
                return prompt[len(prefix) :].strip(" ,.:;-")

        if self._should_web_search(normalized):
            return prompt.strip()
        return ""

    def _looks_like_short_interrupt(self, normalized: str) -> bool:
        words = normalized.split()
        if not words or len(words) > 4:
            return False
        interrupt_words = ("pause", "resume", "cut")
        return any(re.search(rf"\b{word}\b", normalized) for word in interrupt_words)

    def _is_acknowledgement_echo(self, normalized: str) -> bool:
        return normalized == "amy here"

    def _strip_acknowledgement_prefix(self, text: str) -> str:
        pattern = re.compile(r"^\s*(?:amy\s+)?here[\s,.:;-]*", re.IGNORECASE)
        return pattern.sub("", text).strip()

    def _speak_acknowledgement(self) -> None:
        if self.acknowledgment_callback is not None:
            self.acknowledgment_callback()
        self.speaker.speak("Amy here")

    def _emit_acknowledgement(self) -> None:
        if self.acknowledgment_callback is not None:
            self.acknowledgment_callback()

    def _stop_acknowledgement_loop(self) -> None:
        if self.acknowledgment_stop_callback is not None:
            self.acknowledgment_stop_callback()

    def _should_web_search(self, normalized_prompt: str) -> bool:
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
            "who is ",
            "what is ",
            "when is ",
            "where is ",
            "how to ",
            "compare ",
            "tell me about ",
        )
        return any(signal in normalized_prompt for signal in search_signals)

    def _format_web_context(self, query: str, results: list[SearchResult]) -> str:
        if not results:
            return f"Search query: {query}\nNo web results were returned."

        lines = [f"Search query: {query}", "Top web results:"]
        for index, result in enumerate(results, start=1):
            snippet = f" - {result.snippet}" if result.snippet else ""
            content = ""
            if result.content:
                content = f"\n   Extracted text: {result.content[:1000]}"
            lines.append(f"{index}. {result.title}{snippet}{content}")
        return "\n".join(lines)

    def _schedule_idle_timeout_locked(self) -> None:
        self._cancel_idle_timer_locked()
        if self.idle_timeout_seconds <= 0:
            self.status.active_conversation = False
            self.status.phase = AssistantPhase.IDLE
            return

        timer = threading.Timer(self.idle_timeout_seconds, self._set_idle)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _cancel_idle_timer_locked(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _set_idle(self) -> None:
        with self._lock:
            if self.status.paused:
                return
            self.status.active_conversation = False
            self.status.phase = AssistantPhase.IDLE
            self._idle_timer = None

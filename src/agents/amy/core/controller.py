from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
import threading
import time
from typing import Callable, Protocol

from .models import AssistantPhase, AssistantStatus, ConversationTurn, Message
from .prompts import PromptBuilder
from ..memory import MemoryClassifierProtocol, MemoryDecision, MemoryDraft, MemoryStoreProtocol
from ..skills.browser import SearchResult, WebSearcher
from ..runtime.status import AmyStatusReporter


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
    status_reporter: AmyStatusReporter | None = None
    memory_store: MemoryStoreProtocol | None = None
    memory_classifier: MemoryClassifierProtocol | None = None
    web_search: WebSearcher | None = None
    web_search_limit: int = 4
    idle_timeout_seconds: float = 10.0
    follow_up_timeout_seconds: float = 30.0
    usage_logger: Callable[[int, float], None] | None = None
    acknowledgment_callback: Callable[[], None] | None = None
    acknowledgment_stop_callback: Callable[[], None] | None = None
    status: AssistantStatus = field(default_factory=AssistantStatus)
    turns: list[ConversationTurn] = field(default_factory=lambda: list[ConversationTurn]())
    _cancel_event: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _idle_timer: threading.Timer | None = field(default=None, init=False)
    _speech_cooldown_timer: threading.Timer | None = field(default=None, init=False)
    _logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__), init=False, repr=False)
    _acknowledgement_echoes: frozenset[str] = field(
        default_factory=lambda: frozenset({"yes", "yeah", "yep", "no", "ok", "okay", "sure", "right"}),
        init=False,
        repr=False,
    )
    speech_cooldown_seconds: float = 0.6

    def pause(self) -> None:
        with self._lock:
            self._logger.debug("pause requested")
            self._cancel_idle_timer_locked()
            self._cancel_speech_cooldown_locked()
            self._cancel_event.set()
            self.speaker.stop()
            self.status.paused = True
            self.status.phase = AssistantPhase.PAUSED

    def resume(self) -> None:
        with self._lock:
            self._logger.debug("resume requested")
            self.status.paused = False
            if self.status.active_conversation:
                self.status.phase = AssistantPhase.AWAITING_USER_RESPONSE
                self._schedule_post_speech_idle_timeout_locked(expects_follow_up=True)
            else:
                self.status.phase = AssistantPhase.LISTENING

    def cut_channel(self) -> None:
        with self._lock:
            self._logger.debug("cut requested")
            self._cancel_idle_timer_locked()
            self._cancel_speech_cooldown_locked()
            self._cancel_event.set()
            self.speaker.stop()
            self.status.active_conversation = False
            self.status.paused = True
            self.status.phase = AssistantPhase.PAUSED

    def stop(self) -> None:
        with self._lock:
            self._cancel_idle_timer_locked()
            self._cancel_speech_cooldown_locked()
            self._cancel_event.set()
            self.speaker.stop()
            self.status.active_conversation = False
            self.status.phase = AssistantPhase.IDLE
            self.status.paused = False

    def process_transcript(self, transcript: str) -> str | None:
        process_started = time.perf_counter()
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
        if self._is_status_command(normalized) or self._is_status_command(transcript):
            self._logger.debug("status command matched")
            return self._handle_status_check(transcript)

        with self._lock:
            if self.status.paused:
                self._logger.debug("dropping transcript because assistant is paused")
                return None

            if self.status.phase in {AssistantPhase.SPEAKING, AssistantPhase.COOLDOWN}:
                self._logger.debug("dropping transcript because assistant is in speech cooldown")
                return None

            if self.status.active_conversation:
                self._cancel_idle_timer_locked()
            self._cancel_speech_cooldown_locked()

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
                    self.status.phase = AssistantPhase.COOLDOWN
                    self._stop_acknowledgement_loop()
                    self.speaker.speak("Amy here")
                    self._schedule_post_speech_transition_locked(expects_follow_up=True)
                    return None
                self.status.active_conversation = True
                self.status.phase = AssistantPhase.RECORDING
                self._logger.debug("wake word matched; capturing prompt")
            else:
                prompt = self._strip_wake_word(prompt)
                prompt = self._strip_acknowledgement_prefix(prompt)
                if not prompt:
                    self._logger.debug("active conversation but prompt empty after stripping wake word")
                    return None

            explicit_memory_request = self._is_memory_request(prompt)
            memory_decision: MemoryDecision | None = None
            memory_considered = self._should_consider_memory(prompt)
            if self.memory_classifier is not None and memory_considered:
                classifier_started = time.perf_counter()
                try:
                    memory_decision = self.memory_classifier.classify(prompt, self._cancel_event)
                    classifier_elapsed = time.perf_counter() - classifier_started
                    self._logger.debug(
                        "memory classifier decision in %.3fs: save=%s subject=%r confidence=%.2f reason=%r",
                        classifier_elapsed,
                        memory_decision.should_save,
                        memory_decision.subject,
                        memory_decision.confidence,
                        memory_decision.reason,
                    )
                except Exception as exc:  # pragma: no cover - runtime path
                    self._logger.warning("memory classifier failed: %s", exc)
            elif self.memory_classifier is not None:
                self._logger.debug("skipping memory classifier for prompt profile")

            should_save_memory = explicit_memory_request or (
                memory_decision is not None and memory_decision.should_save
            )
            if should_save_memory and self.memory_store is not None:
                self._logger.debug(
                    "memory save triggered: explicit=%s classifier=%s",
                    explicit_memory_request,
                    None if memory_decision is None else memory_decision.should_save,
                )
                draft = self.memory_store.draft_from_prompt(
                    prompt,
                    subject=memory_decision.subject if memory_decision and memory_decision.subject else None,
                )
                if draft is not None:
                    return self._save_memory_draft(draft)

            web_context = ""
            memory_context = ""
            search_query = self._extract_search_query(prompt)
            if self.web_search is not None and search_query:
                web_started = time.perf_counter()
                self._logger.debug("web search triggered: %r", search_query)
                self._emit_acknowledgement()
                web_results = self.web_search.search(search_query, self.web_search_limit)
                web_context = self._format_web_context(search_query, web_results)
                web_elapsed = time.perf_counter() - web_started
                self._logger.debug("web search completed in %.3fs", web_elapsed)
            if self.memory_store is not None:
                memory_started = time.perf_counter()
                memory_context = self.memory_store.retrieve_context(prompt)
                memory_elapsed = time.perf_counter() - memory_started
                self._logger.debug("memory retrieval completed in %.3fs", memory_elapsed)

            self.status.phase = AssistantPhase.THINKING
            self.status.last_user_text = prompt
            self.turns.append(ConversationTurn(role="user", content=prompt))
            self._cancel_event.clear()

        messages = self.prompt_builder.build_messages(
            self.turns[:-1],
            prompt,
            web_context=web_context,
            memory_context=memory_context,
        )
        if self.usage_logger is not None:
            token_count = self._estimate_tokens(messages)
            self.usage_logger(token_count, token_count * 0.00015)
        self._logger.debug("sending %d messages to responder", len(messages))
        reply_started = time.perf_counter()
        reply = self.responder.generate_reply(messages, self._cancel_event)
        reply_elapsed = time.perf_counter() - reply_started
        self._logger.debug("responder completed in %.3fs", reply_elapsed)

        if self._cancel_event.is_set():
            self._logger.debug("reply cancelled before speech")
            return None

        reply = self._limit_follow_up_questions(reply)
        with self._lock:
            self.status.phase = AssistantPhase.SPEAKING
            self.status.last_assistant_text = reply
            self.turns.append(ConversationTurn(role="assistant", content=reply))

        self._logger.debug("speaking reply: %r", reply)
        self._stop_acknowledgement_loop()
        speak_started = time.perf_counter()
        self.speaker.speak(reply)
        speak_elapsed = time.perf_counter() - speak_started

        with self._lock:
            if not self._cancel_event.is_set():
                expects_follow_up = self._reply_expects_follow_up(reply) or self._reply_ends_session_immediately(reply)
                self._logger.debug("reply complete; waiting for follow-up")
                self.status.active_conversation = True
                self._schedule_post_speech_transition_locked(expects_follow_up=expects_follow_up)
        total_elapsed = time.perf_counter() - process_started
        self._logger.debug(
            "transcript profile: total=%.3fs reply=%.3fs speech=%.3fs",
            total_elapsed,
            reply_elapsed,
            speak_elapsed,
        )
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

    def _is_status_command(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        phrases = (
            "status check",
            "check your status",
            "check status",
            "what is your status",
            "whats your status",
            "what's your status",
            "how are you doing",
        )
        return any(phrase in normalized for phrase in phrases)

    def is_interrupt_command(self, transcript: str) -> bool:
        normalized = self._normalize_text(transcript)
        return self._looks_like_short_interrupt(normalized)

    def should_drop_main_transcript(self) -> bool:
        with self._lock:
            return self.status.paused or self.status.phase in {
                AssistantPhase.SPEAKING,
                AssistantPhase.COOLDOWN,
            }

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
        interrupt_words = ("pause", "resume", "cut", "stop")
        return any(re.search(rf"\b{word}\b", normalized) for word in interrupt_words)

    def _reply_expects_follow_up(self, reply: str) -> bool:
        normalized = self._normalize_echo_text(reply)
        if "?" in reply:
            return True

        follow_up_signals = (
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
        return any(signal in normalized for signal in follow_up_signals)

    def _reply_ends_session_immediately(self, reply: str) -> bool:
        normalized = self._normalize_echo_text(reply)
        wrap_up_signals = (
            "anything else i can help you with",
            "anything else i can do",
            "anything else i can help",
            "let me know if you need anything else",
            "let me know if i can help with anything else",
            "is there anything else",
        )
        return any(signal in normalized for signal in wrap_up_signals)

    def _is_memory_request(self, text: str) -> bool:
        request_signals = (
            "remember this",
            "remember that",
            "remember for later",
            "save this for later",
            "don't forget",
            "dont forget",
            "note this",
            "keep this in mind",
        )
        return any(signal in text for signal in request_signals)

    def _should_consider_memory(self, prompt: str) -> bool:
        normalized = self._normalize_text(prompt)
        if self._is_memory_request(normalized):
            return True
        if self._looks_like_direct_question(normalized):
            return False
        if self._extract_search_query(prompt):
            return False

        memory_signals = (
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
        return any(signal in normalized for signal in memory_signals)

    def _looks_like_direct_question(self, normalized: str) -> bool:
        if normalized.endswith("?"):
            return True
        question_starters = (
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
        return normalized.startswith(question_starters)

    def _save_memory_draft(self, draft: MemoryDraft) -> str:
        if self.memory_store is None:
            reply = "I can't save that right now."
        else:
            saved_path = self.memory_store.save_draft(draft)
            reply = f"Saved as `{saved_path.name}`."
        self.status.last_user_text = draft.path.name
        self.status.phase = AssistantPhase.SPEAKING
        self.status.last_assistant_text = reply
        self._stop_acknowledgement_loop()
        self.speaker.speak(reply)
        self.status.active_conversation = True
        self._schedule_post_speech_transition_locked(expects_follow_up=False)
        return reply

    def _handle_status_check(self, transcript: str) -> str:
        reply = self._build_status_report()
        with self._lock:
            self.status.last_user_text = self._strip_wake_word(transcript.strip())
            self.status.phase = AssistantPhase.SPEAKING
            self.status.last_assistant_text = reply
            self.status.active_conversation = True
        self._stop_acknowledgement_loop()
        self.speaker.speak(reply)
        with self._lock:
            self._schedule_post_speech_transition_locked(expects_follow_up=False)
        return reply

    def _build_status_report(self) -> str:
        if self.status_reporter is None:
            status = self.status
            error_text = status.error_message.strip() or "no errors"
            return (
                f"Status check: {status.phase.value}, "
                f"{'paused' if status.paused else 'not paused'}, "
                f"{'active conversation' if status.active_conversation else 'no active conversation'}, "
                f"{error_text}."
            )
        return self.status_reporter.build_report(self.status)

    def _limit_follow_up_questions(self, reply: str) -> str:
        first_question_mark = reply.find("?")
        if first_question_mark < 0:
            return reply.strip()

        if reply.count("?") <= 1:
            return reply.strip()

        return reply[: first_question_mark + 1].strip()

    def _is_acknowledgement_echo(self, normalized: str) -> bool:
        if normalized == "amy here":
            return True
        if not self.status.active_conversation:
            return False

        normalized_echo = self._normalize_echo_text(normalized)
        last_assistant_text = self.status.last_assistant_text
        if not last_assistant_text:
            return False

        normalized_assistant = self._normalize_echo_text(last_assistant_text)
        return (
            normalized_echo in self._acknowledgement_echoes
            and normalized_echo == normalized_assistant
        )

    def _normalize_echo_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

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

    def _schedule_idle_timeout_locked(self, timeout_seconds: float | None = None) -> None:
        self._cancel_idle_timer_locked()
        timeout = self.idle_timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            self.status.active_conversation = False
            self.status.phase = AssistantPhase.IDLE
            return

        timer = threading.Timer(timeout, self._set_idle)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _schedule_post_speech_transition_locked(self, expects_follow_up: bool) -> None:
        self._cancel_speech_cooldown_locked()
        self.status.phase = AssistantPhase.COOLDOWN
        if self.speech_cooldown_seconds <= 0:
            self._speech_cooldown_timer = None
            self.status.phase = (
                AssistantPhase.AWAITING_USER_RESPONSE if expects_follow_up else AssistantPhase.LISTENING
            )
            self._schedule_post_speech_idle_timeout_locked(expects_follow_up)
            return

        timer = threading.Timer(
            self.speech_cooldown_seconds,
            self._finish_post_speech_transition,
            args=(expects_follow_up,),
        )
        timer.daemon = True
        self._speech_cooldown_timer = timer
        timer.start()

    def _cancel_idle_timer_locked(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _cancel_speech_cooldown_locked(self) -> None:
        if self._speech_cooldown_timer is not None:
            self._speech_cooldown_timer.cancel()
            self._speech_cooldown_timer = None

    def _finish_post_speech_transition(self, expects_follow_up: bool) -> None:
        with self._lock:
            if self.status.paused or self.status.phase != AssistantPhase.COOLDOWN:
                return
            self._speech_cooldown_timer = None
            self.status.phase = (
                AssistantPhase.AWAITING_USER_RESPONSE if expects_follow_up else AssistantPhase.LISTENING
            )
            self._schedule_post_speech_idle_timeout_locked(expects_follow_up)

    def _schedule_post_speech_idle_timeout_locked(self, expects_follow_up: bool) -> None:
        if expects_follow_up:
            self._schedule_idle_timeout_locked(timeout_seconds=self.follow_up_timeout_seconds)
            return
        self._schedule_idle_timeout_locked(timeout_seconds=self.idle_timeout_seconds)

    def _set_idle(self) -> None:
        with self._lock:
            if self.status.paused:
                return
            self.status.active_conversation = False
            self.status.phase = AssistantPhase.IDLE
            self._idle_timer = None

__all__ = ["AssistantController", "Responder", "Speaker"]

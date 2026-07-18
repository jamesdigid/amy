from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Callable

from .context.pipeline import ResponsePipeline
from .context.prompts import PromptBuilder
from .memory import MemoryDecision, MemoryDraft
from .models import AssistantStatus, ConversationTurn
from .protocols import MemoryClassifierProtocol, MemoryStoreProtocol, Responder, Speaker, WebSearchProtocol
from .runtime.status import AmyStatusReporter
from .understanding.interpreter import TranscriptInterpreter
from .conversation.session import ConversationSession


@dataclass
class AssistantController:
    prompt_builder: PromptBuilder
    responder: Responder
    speaker: Speaker
    wake_word: str
    status_reporter: AmyStatusReporter | None = None
    memory_store: MemoryStoreProtocol | None = None
    memory_classifier: MemoryClassifierProtocol | None = None
    web_search: WebSearchProtocol | None = None
    web_search_limit: int = 4
    idle_timeout_seconds: float = 10.0
    follow_up_timeout_seconds: float = 30.0
    usage_logger: Callable[[int, float], None] | None = None
    acknowledgment_callback: Callable[[], None] | None = None
    acknowledgment_stop_callback: Callable[[], None] | None = None
    speech_cooldown_seconds: float = 0.6
    _session: ConversationSession = field(init=False, repr=False)
    _interpreter: TranscriptInterpreter = field(init=False, repr=False)
    _pipeline: ResponsePipeline = field(init=False, repr=False)
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._session = ConversationSession()
        self._interpreter = TranscriptInterpreter(wake_word=self.wake_word)
        self._pipeline = ResponsePipeline(
            prompt_builder=self.prompt_builder,
            memory_store=self.memory_store,
            web_search=self.web_search,
            web_search_limit=self.web_search_limit,
            usage_logger=self.usage_logger,
            acknowledgment_callback=self.acknowledgment_callback,
        )

    @property
    def status(self) -> AssistantStatus:
        return self._session.status

    @property
    def turns(self) -> list[ConversationTurn]:
        return self._session.turns

    def pause(self) -> None:
        self._session.pause()
        self.speaker.stop()

    def resume(self) -> None:
        self._session.resume(
            idle_timeout_seconds=self.idle_timeout_seconds,
            follow_up_timeout_seconds=self.follow_up_timeout_seconds,
        )

    def cut_channel(self) -> None:
        self._session.cut_channel()
        self.speaker.stop()

    def stop(self) -> None:
        self._session.stop()
        self.speaker.stop()

    def process_transcript(self, transcript: str) -> str | None:
        process_started = time.perf_counter()
        normalized = self._interpreter.normalize_text(transcript)
        if not normalized:
            return None

        self._logger.debug("received transcript: raw=%r normalized=%r", transcript, normalized)

        if self._interpreter.is_acknowledgement_echo(
            normalized,
            active_conversation=self.status.active_conversation,
            last_assistant_text=self.status.last_assistant_text,
        ):
            self._logger.debug("dropping acknowledgement echo")
            return None

        if self._interpreter.is_pause_command(normalized):
            self._logger.debug("pause command matched")
            self.pause()
            return None
        if self._interpreter.is_resume_command(normalized):
            self._logger.debug("resume command matched")
            self.resume()
            return None
        if self._interpreter.is_cut_command(normalized):
            self._logger.debug("cut command matched")
            self.cut_channel()
            return None
        if self._interpreter.is_status_command(normalized):
            self._logger.debug("status command matched")
            return self._handle_status_check(transcript)

        if self._session.should_drop_main_transcript():
            self._logger.debug("dropping transcript because assistant is in speech cooldown")
            return None

        prompt = transcript.strip()
        prompt = self._interpreter.strip_acknowledgement_prefix(prompt)
        if not self.status.active_conversation:
            if not self._interpreter.starts_with_wake_word(normalized):
                self._logger.debug("dropping transcript because wake word did not match")
                return None
            prompt = self._interpreter.strip_wake_word(prompt)
            prompt = self._interpreter.strip_acknowledgement_prefix(prompt)
            if not prompt:
                self._logger.debug("wake word alone; acknowledging only")
                self._session.acknowledge_wake_word()
                self._stop_acknowledgement_loop()
                self.speaker.speak("Amy here")
                self._session.begin_post_speech(
                    expects_follow_up=True,
                    speech_cooldown_seconds=self.speech_cooldown_seconds,
                    idle_timeout_seconds=self.idle_timeout_seconds,
                    follow_up_timeout_seconds=self.follow_up_timeout_seconds,
                )
                return None
            self._session.begin_recording()
            self._logger.debug("wake word matched; capturing prompt")
        else:
            prompt = self._interpreter.strip_wake_word(prompt)
            prompt = self._interpreter.strip_acknowledgement_prefix(prompt)
            if not prompt:
                self._logger.debug("active conversation but prompt empty after stripping wake word")
                return None

        explicit_memory_request = self._interpreter.is_memory_request(prompt)
        memory_decision: MemoryDecision | None = None
        memory_considered = self._interpreter.should_consider_memory(prompt)
        if self.memory_classifier is not None and memory_considered:
            classifier_started = time.perf_counter()
            try:
                memory_decision = self.memory_classifier.classify(prompt, self._session.cancel_event)
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
                return self._save_memory_draft(draft, prompt)

        web_context, memory_context = self._pipeline.collect_context(prompt, self._interpreter)
        self._session.record_user_turn(prompt)

        messages = self._pipeline.build_messages(
            self.turns[:-1],
            prompt,
            web_context=web_context,
            memory_context=memory_context,
        )
        self._pipeline.log_usage(messages)
        self._logger.debug("sending %d messages to responder", len(messages))
        reply_started = time.perf_counter()
        try:
            reply = self.responder.generate_reply(messages, self._session.cancel_event)
        except Exception as exc:  # pragma: no cover - runtime path
            self._logger.exception("responder failed")
            return self._handle_responder_failure(str(exc))
        reply_elapsed = time.perf_counter() - reply_started
        self._logger.debug("responder completed in %.3fs", reply_elapsed)

        if self._session.cancel_event.is_set():
            self._logger.debug("reply cancelled before speech")
            return None

        reply = self._interpreter.limit_follow_up_questions(reply)
        expects_follow_up = self._interpreter.reply_expects_follow_up(reply) or self._interpreter.reply_ends_session_immediately(reply)
        speak_elapsed = self._deliver_reply(
            reply,
            expects_follow_up=expects_follow_up,
            record_turn=True,
            mark_speaking=True,
        )
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

    def is_interrupt_command(self, transcript: str) -> bool:
        normalized = self._interpreter.normalize_text(transcript)
        return self._interpreter.looks_like_short_interrupt(normalized)

    def should_drop_main_transcript(self) -> bool:
        return self._session.should_drop_main_transcript()

    def _deliver_reply(
        self,
        reply: str,
        *,
        expects_follow_up: bool,
        record_turn: bool,
        mark_speaking: bool,
    ) -> float:
        self._session.record_assistant_reply(
            reply,
            append_turn=record_turn,
            mark_speaking=mark_speaking,
        )
        self._logger.debug("speaking reply: %r", reply)
        self._stop_acknowledgement_loop()
        speak_started = time.perf_counter()
        self.speaker.speak(reply)
        speak_elapsed = time.perf_counter() - speak_started
        if not self._session.cancel_event.is_set():
            self._logger.debug("reply complete; waiting for follow-up")
            self._session.begin_post_speech(
                expects_follow_up=expects_follow_up,
                speech_cooldown_seconds=self.speech_cooldown_seconds,
                idle_timeout_seconds=self.idle_timeout_seconds,
                follow_up_timeout_seconds=self.follow_up_timeout_seconds,
            )
        return speak_elapsed

    def _save_memory_draft(self, draft: MemoryDraft, prompt: str) -> str:
        if self.memory_store is None:
            self._logger.debug("memory save skipped because no memory store is configured")
            return ""

        saved_path = self.memory_store.save_draft(draft)
        self._logger.debug("saved memory draft: %s", saved_path.name)
        reply = "Got it."
        self._session.set_last_user_text(prompt)
        self._deliver_reply(
            reply,
            expects_follow_up=False,
            record_turn=False,
            mark_speaking=False,
        )
        return reply

    def _handle_responder_failure(self, error_message: str) -> str:
        reply = "Sorry, I had trouble reaching the server."
        self._session.set_error_message(error_message)
        self._deliver_reply(
            reply,
            expects_follow_up=False,
            record_turn=True,
            mark_speaking=True,
        )
        return reply

    def _handle_status_check(self, transcript: str) -> str:
        reply = self._build_status_report()
        self._session.set_last_user_text(self._interpreter.strip_wake_word(transcript.strip()))
        self._deliver_reply(
            reply,
            expects_follow_up=False,
            record_turn=True,
            mark_speaking=True,
        )
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

    def _stop_acknowledgement_loop(self) -> None:
        if self.acknowledgment_stop_callback is not None:
            self.acknowledgment_stop_callback()

__all__ = ["AssistantController", "Responder", "Speaker"]

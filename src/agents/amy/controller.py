from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Callable

from .context.pipeline import ResponsePipeline
from .context.prompts import PromptBuilder
from .conversation.session import ConversationSession
from .conversation.side_effects import ConversationSideEffects
from .memory import MemoryDecision
from .models import AssistantStatus, ConversationTurn
from .protocols import MemoryClassifierProtocol, MemoryStoreProtocol, Responder, Speaker, WebSearchProtocol
from .runtime.status import AmyStatusReporter
from .understanding.interpreter import TranscriptInterpreter


@dataclass
class AssistantController:
    prompt_builder: PromptBuilder
    responder: Responder
    speaker: Speaker
    wake_word: str
    status_reporter: object | None = None
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
    _effects: ConversationSideEffects = field(init=False, repr=False)
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
        self._effects = ConversationSideEffects(
            speaker=self.speaker,
            memory_store=self.memory_store,
            status_reporter=self.status_reporter,
            acknowledgment_stop_callback=self.acknowledgment_stop_callback,
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
            return self._effects.handle_status_check(
                transcript,
                interpreter=self._interpreter,
                session=self._session,
                idle_timeout_seconds=self.idle_timeout_seconds,
                follow_up_timeout_seconds=self.follow_up_timeout_seconds,
                speech_cooldown_seconds=self.speech_cooldown_seconds,
                logger=self._logger,
            )

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
                self._effects.stop_acknowledgement_loop()
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
                return self._effects.save_memory_draft(
                    draft,
                    prompt,
                    session=self._session,
                    idle_timeout_seconds=self.idle_timeout_seconds,
                    follow_up_timeout_seconds=self.follow_up_timeout_seconds,
                    speech_cooldown_seconds=self.speech_cooldown_seconds,
                    logger=self._logger,
                )

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
            return self._effects.handle_responder_failure(
                str(exc),
                session=self._session,
                idle_timeout_seconds=self.idle_timeout_seconds,
                follow_up_timeout_seconds=self.follow_up_timeout_seconds,
                speech_cooldown_seconds=self.speech_cooldown_seconds,
                logger=self._logger,
            )
        reply_elapsed = time.perf_counter() - reply_started
        self._logger.debug("responder completed in %.3fs", reply_elapsed)

        if self._session.cancel_event.is_set():
            self._logger.debug("reply cancelled before speech")
            return None

        reply = self._interpreter.limit_follow_up_questions(reply)
        expects_follow_up = self._interpreter.reply_expects_follow_up(reply) or self._interpreter.reply_ends_session_immediately(reply)
        speak_elapsed = self._effects.deliver_reply(
            reply,
            expects_follow_up=expects_follow_up,
            record_turn=True,
            mark_speaking=True,
            session=self._session,
            idle_timeout_seconds=self.idle_timeout_seconds,
            follow_up_timeout_seconds=self.follow_up_timeout_seconds,
            speech_cooldown_seconds=self.speech_cooldown_seconds,
            logger=self._logger,
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

__all__ = ["AssistantController", "Responder", "Speaker"]

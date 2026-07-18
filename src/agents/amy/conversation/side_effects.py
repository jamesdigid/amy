from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Callable

from .session import ConversationSession
from ..memory import MemoryDraft
from ..models import AssistantStatus
from ..protocols import MemoryStoreProtocol, Speaker
from ..runtime.status import AmyStatusReporter
from ..understanding.interpreter import TranscriptInterpreter


@dataclass
class ConversationSideEffects:
    speaker: Speaker
    memory_store: MemoryStoreProtocol | None = None
    status_reporter: AmyStatusReporter | None = None
    acknowledgment_stop_callback: Callable[[], None] | None = None

    def deliver_reply(
        self,
        reply: str,
        *,
        expects_follow_up: bool,
        record_turn: bool,
        mark_speaking: bool,
        session: ConversationSession,
        idle_timeout_seconds: float,
        follow_up_timeout_seconds: float,
        speech_cooldown_seconds: float,
        logger: logging.Logger,
    ) -> float:
        session.record_assistant_reply(
            reply,
            append_turn=record_turn,
            mark_speaking=mark_speaking,
        )
        logger.debug("speaking reply: %r", reply)
        self.stop_acknowledgement_loop()
        speak_started = time.perf_counter()
        self.speaker.speak(reply)
        speak_elapsed = time.perf_counter() - speak_started
        if not session.cancel_event.is_set():
            logger.debug("reply complete; waiting for follow-up")
            session.begin_post_speech(
                expects_follow_up=expects_follow_up,
                speech_cooldown_seconds=speech_cooldown_seconds,
                idle_timeout_seconds=idle_timeout_seconds,
                follow_up_timeout_seconds=follow_up_timeout_seconds,
            )
        return speak_elapsed

    def save_memory_draft(
        self,
        draft: MemoryDraft,
        prompt: str,
        *,
        session: ConversationSession,
        idle_timeout_seconds: float,
        follow_up_timeout_seconds: float,
        speech_cooldown_seconds: float,
        logger: logging.Logger,
    ) -> str:
        if self.memory_store is None:
            logger.debug("memory save skipped because no memory store is configured")
            return ""

        saved_path = self.memory_store.save_draft(draft)
        logger.debug("saved memory draft: %s", saved_path.name)
        reply = "Got it."
        session.set_last_user_text(prompt)
        self.deliver_reply(
            reply,
            expects_follow_up=False,
            record_turn=False,
            mark_speaking=False,
            session=session,
            idle_timeout_seconds=idle_timeout_seconds,
            follow_up_timeout_seconds=follow_up_timeout_seconds,
            speech_cooldown_seconds=speech_cooldown_seconds,
            logger=logger,
        )
        return reply

    def handle_responder_failure(
        self,
        error_message: str,
        *,
        session: ConversationSession,
        idle_timeout_seconds: float,
        follow_up_timeout_seconds: float,
        speech_cooldown_seconds: float,
        logger: logging.Logger,
    ) -> str:
        reply = "Sorry, I had trouble reaching the server."
        session.set_error_message(error_message)
        self.deliver_reply(
            reply,
            expects_follow_up=False,
            record_turn=True,
            mark_speaking=True,
            session=session,
            idle_timeout_seconds=idle_timeout_seconds,
            follow_up_timeout_seconds=follow_up_timeout_seconds,
            speech_cooldown_seconds=speech_cooldown_seconds,
            logger=logger,
        )
        return reply

    def handle_status_check(
        self,
        transcript: str,
        *,
        interpreter: TranscriptInterpreter,
        session: ConversationSession,
        idle_timeout_seconds: float,
        follow_up_timeout_seconds: float,
        speech_cooldown_seconds: float,
        logger: logging.Logger,
    ) -> str:
        reply = self.build_status_report(session.status)
        session.set_last_user_text(interpreter.strip_wake_word(transcript.strip()))
        self.deliver_reply(
            reply,
            expects_follow_up=False,
            record_turn=True,
            mark_speaking=True,
            session=session,
            idle_timeout_seconds=idle_timeout_seconds,
            follow_up_timeout_seconds=follow_up_timeout_seconds,
            speech_cooldown_seconds=speech_cooldown_seconds,
            logger=logger,
        )
        return reply

    def build_status_report(self, status: AssistantStatus) -> str:
        if self.status_reporter is None:
            error_text = status.error_message.strip() or "no errors"
            return (
                f"Status check: {status.phase.value}, "
                f"{'paused' if status.paused else 'not paused'}, "
                f"{'active conversation' if status.active_conversation else 'no active conversation'}, "
                f"{error_text}."
            )
        return self.status_reporter.build_report(status)

    def stop_acknowledgement_loop(self) -> None:
        if self.acknowledgment_stop_callback is not None:
            self.acknowledgment_stop_callback()

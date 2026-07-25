from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading

from ..models import AssistantPhase, AssistantStatus, ConversationTurn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestGeneration:
    request_id: int
    cancel_event: threading.Event


@dataclass
class ConversationSession:
    status: AssistantStatus = field(default_factory=AssistantStatus)
    turns: list[ConversationTurn] = field(default_factory=list[ConversationTurn])
    _cancel_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _idle_timer: threading.Timer | None = field(default=None, init=False, repr=False)
    _speech_cooldown_timer: threading.Timer | None = field(default=None, init=False, repr=False)
    _epoch: int = field(default=0, init=False, repr=False)
    _request_generation: RequestGeneration | None = field(default=None, init=False, repr=False)

    @property
    def cancel_event(self) -> threading.Event:
        with self._lock:
            if self._request_generation is not None:
                return self._request_generation.cancel_event
            return self._cancel_event

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def begin_processing(self) -> RequestGeneration:
        with self._lock:
            self._cancel_idle_timer_locked()
            self._cancel_speech_cooldown_locked()
            self._cancel_active_generation_locked()
            self._epoch += 1
            generation = RequestGeneration(
                request_id=self._epoch,
                cancel_event=threading.Event(),
            )
            self._request_generation = generation
            self.status.phase = AssistantPhase.THINKING
            return generation

    def claim_delivery(self, request_id: int) -> bool:
        with self._lock:
            generation = self._request_generation
            if generation is None or generation.request_id != request_id:
                return False
            if self.status.paused or generation.cancel_event.is_set():
                return False
            self.status.phase = AssistantPhase.SPEAKING
            return True

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def pause(self) -> None:
        with self._lock:
            logger.debug("pause requested")
            self._cancel_idle_timer_locked()
            self._cancel_speech_cooldown_locked()
            self._cancel_active_generation_locked()
            self._epoch += 1
            self.status.paused = True
            self.status.phase = AssistantPhase.PAUSED

    def resume(self, *, idle_timeout_seconds: float, follow_up_timeout_seconds: float) -> None:
        with self._lock:
            logger.debug("resume requested")
            self._cancel_active_generation_locked()
            self._epoch += 1
            self.status.paused = False
            if self.status.active_conversation:
                self.status.phase = AssistantPhase.AWAITING_USER_RESPONSE
                self._schedule_post_speech_idle_timeout_locked(
                    expects_follow_up=True,
                    idle_timeout_seconds=idle_timeout_seconds,
                    follow_up_timeout_seconds=follow_up_timeout_seconds,
                )
            else:
                self.status.phase = AssistantPhase.LISTENING

    def stop(self) -> None:
        with self._lock:
            self._cancel_idle_timer_locked()
            self._cancel_speech_cooldown_locked()
            self._cancel_active_generation_locked()
            self._epoch += 1
            self.status.active_conversation = False
            self.status.phase = AssistantPhase.IDLE
            self.status.paused = False

    def begin_recording(self) -> None:
        with self._lock:
            self.status.active_conversation = True
            self.status.phase = AssistantPhase.RECORDING

    def acknowledge_wake_word(self) -> None:
        with self._lock:
            self.status.active_conversation = True
            self.status.phase = AssistantPhase.COOLDOWN

    def record_user_turn(self, prompt: str) -> None:
        with self._lock:
            self.status.last_user_text = prompt
            self.turns.append(ConversationTurn(role="user", content=prompt))

    def set_last_user_text(self, prompt: str) -> None:
        with self._lock:
            self.status.last_user_text = prompt

    def set_error_message(self, error_message: str) -> None:
        with self._lock:
            self.status.error_message = error_message

    def record_assistant_reply(
        self,
        reply: str,
        *,
        append_turn: bool = True,
        mark_speaking: bool = True,
    ) -> None:
        with self._lock:
            if mark_speaking:
                self.status.phase = AssistantPhase.SPEAKING
            self.status.last_assistant_text = reply
            self.status.active_conversation = True
            if append_turn:
                self.turns.append(ConversationTurn(role="assistant", content=reply))

    def begin_post_speech(
        self,
        *,
        expects_follow_up: bool,
        speech_cooldown_seconds: float,
        idle_timeout_seconds: float,
        follow_up_timeout_seconds: float,
    ) -> None:
        with self._lock:
            self._schedule_post_speech_transition_locked(
                expects_follow_up=expects_follow_up,
                speech_cooldown_seconds=speech_cooldown_seconds,
                idle_timeout_seconds=idle_timeout_seconds,
                follow_up_timeout_seconds=follow_up_timeout_seconds,
            )

    def should_drop_main_transcript(self) -> bool:
        with self._lock:
            return self.status.paused or self.status.phase in {
                AssistantPhase.SPEAKING,
                AssistantPhase.COOLDOWN,
                AssistantPhase.THINKING,
            }

    def _schedule_idle_timeout_locked(self, timeout_seconds: float) -> None:
        self._cancel_idle_timer_locked()
        if timeout_seconds <= 0:
            self.status.active_conversation = False
            self.status.phase = AssistantPhase.IDLE
            return

        timer = threading.Timer(timeout_seconds, self._set_idle)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _schedule_post_speech_transition_locked(
        self,
        *,
        expects_follow_up: bool,
        speech_cooldown_seconds: float,
        idle_timeout_seconds: float,
        follow_up_timeout_seconds: float,
    ) -> None:
        self._cancel_speech_cooldown_locked()
        self.status.phase = AssistantPhase.COOLDOWN
        if speech_cooldown_seconds <= 0:
            self._speech_cooldown_timer = None
            self.status.phase = (
                AssistantPhase.AWAITING_USER_RESPONSE if expects_follow_up else AssistantPhase.LISTENING
            )
            self._schedule_post_speech_idle_timeout_locked(
                expects_follow_up=expects_follow_up,
                idle_timeout_seconds=idle_timeout_seconds,
                follow_up_timeout_seconds=follow_up_timeout_seconds,
            )
            return

        timer = threading.Timer(
            speech_cooldown_seconds,
            self._finish_post_speech_transition,
            args=(expects_follow_up, idle_timeout_seconds, follow_up_timeout_seconds),
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

    def _cancel_active_generation_locked(self) -> None:
        generation = self._request_generation
        if generation is not None:
            generation.cancel_event.set()
            self._request_generation = None

    def _finish_post_speech_transition(
        self,
        expects_follow_up: bool,
        idle_timeout_seconds: float,
        follow_up_timeout_seconds: float,
    ) -> None:
        with self._lock:
            if self.status.paused or self.status.phase != AssistantPhase.COOLDOWN:
                return
            self._speech_cooldown_timer = None
            self.status.phase = (
                AssistantPhase.AWAITING_USER_RESPONSE if expects_follow_up else AssistantPhase.LISTENING
            )
            self._schedule_post_speech_idle_timeout_locked(
                expects_follow_up=expects_follow_up,
                idle_timeout_seconds=idle_timeout_seconds,
                follow_up_timeout_seconds=follow_up_timeout_seconds,
            )

    def _schedule_post_speech_idle_timeout_locked(
        self,
        *,
        expects_follow_up: bool,
        idle_timeout_seconds: float,
        follow_up_timeout_seconds: float,
    ) -> None:
        timeout_seconds = follow_up_timeout_seconds if expects_follow_up else idle_timeout_seconds
        self._schedule_idle_timeout_locked(timeout_seconds=timeout_seconds)

    def _set_idle(self) -> None:
        with self._lock:
            if self.status.paused:
                return
            self.status.active_conversation = False
            self.status.phase = AssistantPhase.IDLE
            self._idle_timer = None

__all__ = ["ConversationSession"]

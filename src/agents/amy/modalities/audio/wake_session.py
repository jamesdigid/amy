from __future__ import annotations

from dataclasses import dataclass, field

from .models import AudioConfig
from .speech_tracker import SpeechTracker


@dataclass(slots=True)
class WakeSession:
    """Tracks per-wake-event audio state and wake-attempt state."""

    config: AudioConfig
    tracker: SpeechTracker = field(init=False)
    attempted: bool = field(default=False, init=False)
    matched: bool = field(default=False, init=False)
    _frames_since_attempt: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.tracker = SpeechTracker(
            hold_frames=self.config.wake_hold_frames,
            window_frames=self.config.wake_window_frames,
        )

    def observe(self, *, sequence: int, is_speech: bool, energy: float = 0.0) -> None:
        self._frames_since_attempt += 1
        self.tracker.observe(sequence=sequence, is_speech=is_speech, energy=energy)
        if self.tracker.speech_started:
            self.attempted = False
            self.matched = False
            # Let the first valid speech window attempt immediately; later misses
            # are spaced by wake_poll_frames so a partial "a" can be retried as "amy".
            self._frames_since_attempt = self.config.wake_poll_frames

    def should_attempt(self) -> bool:
        return (
            self.tracker.speech_active
            and not self.matched
            and self.tracker.consecutive_speech_frames >= self.config.wake_min_speech_frames
            and self._frames_since_attempt >= self.config.wake_poll_frames
        )

    def should_attempt_trailing(self) -> bool:
        return self.tracker.speech_finished and self.attempted and not self.matched

    def record_attempt(self, *, matched: bool) -> None:
        self.attempted = True
        self.matched = matched
        self._frames_since_attempt = 0

    def reset(self) -> None:
        self.tracker.reset()
        self.attempted = False
        self.matched = False
        self._frames_since_attempt = 0

    @property
    def speech_active(self) -> bool:
        return self.tracker.speech_active

    @property
    def speech_finished(self) -> bool:
        return self.tracker.speech_finished

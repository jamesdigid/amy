from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class _SpeechFrame:
    sequence: int
    is_speech: bool
    energy: float = 0.0


@dataclass(slots=True)
class SpeechTracker:
    """Tracks speech-event edges, hold timing, and rolling speech history."""

    hold_frames: int
    window_frames: int | None = None
    _speech_active: bool = field(default=False, init=False, repr=False)
    _speech_started: bool = field(default=False, init=False, repr=False)
    _speech_finished: bool = field(default=False, init=False, repr=False)
    _speech_frames: int = field(default=0, init=False, repr=False)
    _consecutive_speech_frames: int = field(default=0, init=False, repr=False)
    _frames_since_speech: int = field(default=0, init=False, repr=False)
    _window: deque[_SpeechFrame] = field(init=False, repr=False)
    _window_speech_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._window = deque(maxlen=max(0, self.window_frames))

    def observe(self, *, is_speech: bool, sequence: int = 0, energy: float = 0.0) -> None:
        self._speech_started = False
        self._speech_finished = False
        self._observe_window(sequence=sequence, is_speech=is_speech, energy=energy)

        if is_speech:
            self._speech_frames += 1
            self._consecutive_speech_frames += 1
            self._frames_since_speech = 0
            if not self._speech_active:
                self._speech_active = True
                self._speech_started = True
            return

        self._consecutive_speech_frames = 0
        if not self._speech_active:
            return

        self._frames_since_speech += 1
        if self._frames_since_speech > self.hold_frames:
            self._speech_active = False
            self._speech_finished = True
            self._speech_frames = 0
            self._consecutive_speech_frames = 0
            self._frames_since_speech = 0

    def reset(self) -> None:
        self._speech_active = False
        self._speech_started = False
        self._speech_finished = False
        self._speech_frames = 0
        self._consecutive_speech_frames = 0
        self._frames_since_speech = 0
        self._window.clear()
        self._window_speech_count = 0

    def _observe_window(self, *, sequence: int, is_speech: bool, energy: float) -> None:
        if self.window_frames is None:
            return
        if len(self._window) == self._window.maxlen and self._window[0].is_speech:
            self._window_speech_count -= 1
        self._window.append(_SpeechFrame(sequence=sequence, is_speech=is_speech, energy=energy))
        if is_speech:
            self._window_speech_count += 1

    @property
    def speech_started(self) -> bool:
        return self._speech_started

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    @property
    def speech_finished(self) -> bool:
        return self._speech_finished

    @property
    def speech_frames(self) -> int:
        return self._speech_frames

    @property
    def consecutive_speech_frames(self) -> int:
        return self._consecutive_speech_frames

    @property
    def frame_count(self) -> int:
        return len(self._window)

    @property
    def window_full(self) -> bool:
        return self.window_frames > 0 and self.frame_count == self.window_frames

    @property
    def speech_density(self) -> float:
        if not self._window:
            return 0.0
        return self._window_speech_count / self.frame_count

    @property
    def window_speech_frames(self) -> int:
        return self._window_speech_count

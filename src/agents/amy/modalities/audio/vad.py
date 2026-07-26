from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import AudioConfig, AudioFrame


@dataclass(frozen=True, slots=True)
class VadState:
    is_speech: bool
    silence_frames: int
    speech_frames: int


class SpeechDetector(Protocol):
    def observe(self, frame: AudioFrame) -> VadState: ...

    def reset(self) -> None: ...


class EnergyVad:
    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._speaking = False
        self._silence_count = 0
        self._speech_count = 0

    def _is_speech(self, frame: AudioFrame) -> bool:
        return frame.rms >= self._config.rms_threshold

    def observe(self, frame: AudioFrame) -> VadState:
        is_speech = self._is_speech(frame)
        if is_speech:
            self._speaking = True
            self._silence_count = 0
            self._speech_count += 1
        else:
            self._silence_count += 1
            if not self._speaking:
                self._speech_count = 0
        return VadState(
            is_speech=is_speech,
            silence_frames=self._silence_count,
            speech_frames=self._speech_count,
        )

    def reset(self) -> None:
        self._speaking = False
        self._silence_count = 0
        self._speech_count = 0


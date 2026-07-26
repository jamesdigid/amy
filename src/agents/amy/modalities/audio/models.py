from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math


def _rms_stdlib(data: bytes) -> int:
    from array import array

    samples = array("h")
    samples.frombytes(data)
    if not samples:
        return 0
    return math.isqrt(sum(sample * sample for sample in samples) // len(samples))


def _resolve_rms() -> Callable[[bytes], int]:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy ships with the audio extra
        return _rms_stdlib

    def _rms_numpy(data: bytes) -> int:
        samples = np.frombuffer(data, dtype="<i2").astype(np.int64)
        if samples.size == 0:
            return 0
        return math.isqrt(int(samples @ samples) // samples.size)

    return _rms_numpy


_compute_rms = _resolve_rms()


@dataclass(slots=True)
class AudioFrame:
    data: bytes
    overflow: bool = False
    sequence: int = 0
    _rms: int | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def rms(self) -> int:
        """Root-mean-square amplitude, computed once and shared by every consumer.

        Both the capture and wake detectors observe the same frames, so caching
        here halves the work. Two threads racing to populate it is harmless
        because the computation is pure.
        """
        if self._rms is None:
            self._rms = _compute_rms(self.data)
        return self._rms


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    silence_ms: int = 700
    rms_threshold: int = 500
    wake_rms_threshold: int = 650
    max_utterance_ms: int = 30_000
    input_device: str | int | None = None
    ring_buffer_ms: int = 2000
    wake_window_ms: int = 1200
    wake_min_window_ms: int = 500
    wake_poll_ms: int = 300
    wake_min_speech_ms: int = 150
    wake_hold_ms: int = 450
    wake_cooldown_ms: int = 1500
    wake_model_repo: str = "mlx-community/whisper-tiny"

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def silence_frames(self) -> int:
        return max(1, int(self.silence_ms / self.frame_ms))

    @property
    def max_utterance_frames(self) -> int:
        return max(1, int(self.max_utterance_ms / self.frame_ms))

    @property
    def ring_buffer_frames(self) -> int:
        return max(1, int(self.ring_buffer_ms / self.frame_ms))

    @property
    def wake_window_frames(self) -> int:
        return max(1, int(self.wake_window_ms / self.frame_ms))

    @property
    def wake_min_window_frames(self) -> int:
        return max(1, int(self.wake_min_window_ms / self.frame_ms))

    @property
    def wake_min_speech_frames(self) -> int:
        return max(1, int(self.wake_min_speech_ms / self.frame_ms))

    @property
    def wake_poll_frames(self) -> int:
        return max(1, int(self.wake_poll_ms / self.frame_ms))

    @property
    def wake_hold_frames(self) -> int:
        return max(1, int(self.wake_hold_ms / self.frame_ms))


@dataclass(slots=True)
class AudioSegment:
    pcm: bytes
    duration_seconds: float


__all__ = ["AudioConfig", "AudioFrame", "AudioSegment"]

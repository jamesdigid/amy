from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading

from .models import AudioConfig, AudioFrame


@dataclass(frozen=True, slots=True)
class RingSnapshot:
    frames: tuple[AudioFrame, ...]
    last_sequence: int

    @property
    def pcm(self) -> bytes:
        return b"".join(frame.data for frame in self.frames)

    @property
    def rms(self) -> int:
        total_samples = 0
        total_square = 0
        for frame in self.frames:
            samples = len(frame.data) // 2
            if samples == 0:
                continue
            total_samples += samples
            total_square += frame.rms * frame.rms * samples
        if total_samples == 0:
            return 0
        return math.isqrt(total_square // total_samples)


class AudioRingBuffer:
    def __init__(self, config: AudioConfig, capacity_ms: int | None = None) -> None:
        self._config = config
        self._capacity_frames = max(1, int((capacity_ms or config.ring_buffer_ms) / config.frame_ms))
        self._frames: deque[AudioFrame] = deque(maxlen=self._capacity_frames)
        self._lock = threading.Lock()

    def append(self, frame: AudioFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    def snapshot(self, duration_ms: int | None = None) -> RingSnapshot:
        with self._lock:
            frames = tuple(self._frames)

        if duration_ms is None:
            selected = frames
        else:
            keep_frames = max(1, int(duration_ms / self._config.frame_ms))
            selected = frames[-keep_frames:]

        last_sequence = selected[-1].sequence if selected else 0
        return RingSnapshot(frames=selected, last_sequence=last_sequence)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()


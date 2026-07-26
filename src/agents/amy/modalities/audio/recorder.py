from __future__ import annotations

from dataclasses import dataclass, field

from .models import AudioConfig, AudioFrame, AudioSegment
from .ring_buffer import RingSnapshot
from .vad import VadState


def _empty_audio_frame_list() -> list[AudioFrame]:
    return []


@dataclass
class CommandRecorder:
    config: AudioConfig
    _frames: list[AudioFrame] = field(default_factory=_empty_audio_frame_list, init=False, repr=False)
    _recording: bool = field(default=False, init=False, repr=False)
    _silence_count: int = field(default=0, init=False, repr=False)
    _speech_count: int = field(default=0, init=False, repr=False)
    _last_sequence: int = field(default=0, init=False, repr=False)

    def begin(self, snapshot: RingSnapshot) -> None:
        self._frames = list(snapshot.frames)
        self._recording = True
        self._silence_count = 0
        self._speech_count = len(self._frames)
        self._last_sequence = snapshot.last_sequence

    def feed(self, frame: AudioFrame, vad_state: VadState) -> AudioSegment | None:
        if not self._recording:
            return None
        if frame.sequence <= self._last_sequence:
            return None

        self._frames.append(frame)
        self._last_sequence = frame.sequence
        self._speech_count += 1
        if vad_state.is_speech:
            self._silence_count = 0
        else:
            self._silence_count += 1

        if self._silence_count < self.config.silence_frames and self._speech_count < self.config.max_utterance_frames:
            return None

        pcm = b"".join(frame.data for frame in self._frames)
        frame_count = len(pcm) // 2
        duration_seconds = frame_count / self.config.sample_rate
        self.reset()
        return AudioSegment(pcm=pcm, duration_seconds=duration_seconds)

    def reset(self) -> None:
        self._frames = []
        self._recording = False
        self._silence_count = 0
        self._speech_count = 0
        self._last_sequence = 0

    @property
    def is_recording(self) -> bool:
        return self._recording


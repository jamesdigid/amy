from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import audioop
import contextlib
from collections import deque
from typing import Deque, Iterator, Protocol, cast
import wave


class _InputStream(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def read(self, frames: int) -> tuple[bytes, bool]: ...


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    pre_roll_ms: int = 300
    silence_ms: int = 700
    rms_threshold: int = 500

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def pre_roll_frames(self) -> int:
        return max(1, int(self.pre_roll_ms / self.frame_ms))

    @property
    def silence_frames(self) -> int:
        return max(1, int(self.silence_ms / self.frame_ms))


@dataclass
class AudioSegment:
    path: Path
    duration_seconds: float


class SpeechSegmenter:
    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._pre_roll: Deque[bytes] = deque(maxlen=config.pre_roll_frames)
        self._speech_frames: list[bytes] = []
        self._speaking = False
        self._silence_count = 0

    def feed(self, frame: bytes) -> AudioSegment | None:
        self._pre_roll.append(frame)
        rms = audioop.rms(frame, 2)

        if not self._speaking:
            if rms >= self._config.rms_threshold:
                self._speaking = True
                self._speech_frames = list(self._pre_roll)
                self._silence_count = 0
            return None

        self._speech_frames.append(frame)
        if rms < self._config.rms_threshold:
            self._silence_count += 1
        else:
            self._silence_count = 0

        if self._silence_count < self._config.silence_frames:
            return None

        audio = b"".join(self._speech_frames)
        self._speech_frames = []
        self._speaking = False
        self._silence_count = 0
        return self._write_temp_wav(audio)

    def _write_temp_wav(self, audio: bytes) -> AudioSegment:
        import tempfile

        frame_count = len(audio) // 2
        duration_seconds = frame_count / self._config.sample_rate
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        with contextlib.closing(wave.open(str(temp_path), "wb")) as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._config.sample_rate)
            wav_file.writeframes(audio)
        return AudioSegment(path=temp_path, duration_seconds=duration_seconds)

    def reset(self) -> None:
        self._pre_roll.clear()
        self._speech_frames = []
        self._speaking = False
        self._silence_count = 0


class MicrophoneSource:
    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._stream: _InputStream | None = None

    def __enter__(self) -> "MicrophoneSource":
        import sounddevice as sd

        stream = cast(
            _InputStream,
            sd.RawInputStream(
                channels=1,
                samplerate=self._config.sample_rate,
                dtype="int16",
                blocksize=self._config.frame_samples,
            ),
        )
        self._stream = stream
        stream.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def frames(self) -> Iterator[bytes]:
        if self._stream is None:
            raise RuntimeError("MicrophoneSource must be entered before reading frames")

        while True:
            data, _overflow = self._stream.read(self._config.frame_samples)
            yield bytes(data)

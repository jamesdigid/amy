from __future__ import annotations

from array import array
import importlib
from dataclasses import dataclass, field
from pathlib import Path
import contextlib
from collections import deque
import math
import sys
from typing import Deque, Iterator, Protocol, TypedDict, cast
import logging
import platform
import subprocess
import threading
import wave

logger = logging.getLogger(__name__)


class _InputStream(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def read(self, frames: int) -> tuple[bytes, bool]: ...


class _TtsEngine(Protocol):
    def setProperty(self, name: str, value: object) -> None: ...

    def say(self, text: str) -> None: ...

    def runAndWait(self) -> None: ...


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
        rms = self._rms(frame)

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

    def _rms(self, frame: bytes) -> int:
        if not frame:
            return 0

        samples = array("h")
        samples.frombytes(frame)
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            return 0

        mean_square = sum(sample * sample for sample in samples) / len(samples)
        return math.isqrt(int(mean_square))

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


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> str: ...


class _MlxWhisperTranscribeResult(TypedDict):
    text: str


class _MlxWhisperModule(Protocol):
    def transcribe(
        self,
        audio: object,
        *,
        path_or_hf_repo: str,
        language: str | None = None,
        verbose: bool | None = None,
        temperature: float = 0.0,
        condition_on_previous_text: bool = False,
        word_timestamps: bool = False,
    ) -> _MlxWhisperTranscribeResult: ...


@dataclass
class MlxWhisperTranscriber:
    model_repo: str = "mlx-community/whisper-large-v3-turbo"
    language: str | None = None
    _model_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._module: _MlxWhisperModule | None = None
        self._warmed = False

    def _load_module(self) -> _MlxWhisperModule:
        if self._module is not None:
            return self._module

        with self._model_lock:
            if self._module is None:
                self._module = cast(_MlxWhisperModule, importlib.import_module("mlx_whisper"))
        return self._module

    def _transcribe_audio(self, audio: object) -> _MlxWhisperTranscribeResult:
        module = self._load_module()
        return module.transcribe(
            audio,
            path_or_hf_repo=self.model_repo,
            language=self.language,
            verbose=False,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
        )

    def warmup(self) -> None:
        if self._warmed:
            return

        module = self._load_module()
        with self._model_lock:
            if self._warmed:
                return

            import numpy as np

            silence = np.zeros(16000, dtype=np.float32)
            module.transcribe(
                silence,
                path_or_hf_repo=self.model_repo,
                language=self.language,
                verbose=False,
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            self._warmed = True

    def transcribe(self, audio_path: Path) -> str:
        result = self._transcribe_audio(str(audio_path))
        return result["text"].strip()


@dataclass
class StubTranscriber:
    transcript: str

    def transcribe(self, audio_path: Path) -> str:  # noqa: ARG002
        return self.transcript


@dataclass
class LocalSpeaker:
    voice: str | None = None
    _process: subprocess.Popen[str] | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    is_speaking: threading.Event = field(default_factory=threading.Event, init=False)

    def speak(self, text: str) -> None:
        logger.debug("tts speak requested: %r", text)
        self.stop()
        self.is_speaking.set()
        try:
            if platform.system() == "Darwin":
                command = ["say"]
                if self.voice:
                    command.extend(["-v", self.voice])
                command.append(text)
                process = subprocess.Popen(command, text=True)
                with self._lock:
                    self._process = process
                process.wait()
                return

            try:
                import pyttsx3  # pyright: ignore[reportMissingImports]

                engine = cast(_TtsEngine, pyttsx3.init())  # pyright: ignore[reportUnknownMemberType]
                if self.voice:
                    engine.setProperty("voice", self.voice)
                engine.say(text)
                engine.runAndWait()
            except Exception:
                print(text)
        finally:
            with self._lock:
                self._process = None
            self.is_speaking.clear()

    def stop(self) -> None:
        logger.debug("tts stop requested")
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                self._process.wait(timeout=1)
            self._process = None
            self.is_speaking.clear()

__all__ = [
    "AudioConfig",
    "AudioSegment",
    "LocalSpeaker",
    "MicrophoneSource",
    "MlxWhisperTranscriber",
    "SpeechSegmenter",
    "StubTranscriber",
    "Transcriber",
]

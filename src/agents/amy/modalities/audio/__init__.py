from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import audioop
import contextlib
from collections import deque
from typing import Deque, Iterable, Iterator, Protocol, cast
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


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> str: ...


class _WhisperSegment(Protocol):
    @property
    def text(self) -> str: ...


class _WhisperModel(Protocol):
    def transcribe(
        self,
        audio: str,
        *,
        language: str | None,
        beam_size: int,
        vad_filter: bool,
    ) -> tuple[Iterable[_WhisperSegment], object]: ...


@dataclass
class FasterWhisperTranscriber:
    model_name: str = "base"
    language: str | None = None

    def __post_init__(self) -> None:
        self._model: _WhisperModel | None = None

    def _load_model(self) -> _WhisperModel:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = cast(
                _WhisperModel,
                WhisperModel(self.model_name, device="cpu", compute_type="int8"),
            )
        return self._model

    def transcribe(self, audio_path: Path) -> str:
        model = self._load_model()
        segments, _info = model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=1,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text


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
    "FasterWhisperTranscriber",
    "LocalSpeaker",
    "MicrophoneSource",
    "SpeechSegmenter",
    "StubTranscriber",
    "Transcriber",
]

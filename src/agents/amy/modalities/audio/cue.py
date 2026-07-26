from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import platform
import subprocess
import tempfile
import threading
import wave
from typing import Protocol, cast

from .models import AudioConfig

logger = logging.getLogger(__name__)

_SAMPLE_WIDTH = 2


class _OutputBuffer(Protocol):
    def __setitem__(self, index: slice, value: bytes) -> None: ...


class _OutputStream(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


OutputStreamFactory = Callable[..., _OutputStream]


def render_speech_pcm(
    text: str,
    *,
    sample_rate: int,
    voice: str | None = None,
    timeout: float = 15.0,
) -> bytes | None:
    """Render ``text`` to PCM once with macOS ``say``."""
    if platform.system() != "Darwin":
        return None

    command = ["say", f"--data-format=LEI16@{sample_rate}"]
    if voice:
        command.extend(["-v", voice])

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "cue.wav"
            subprocess.run(
                [*command, "-o", str(target), text],
                check=True,
                timeout=timeout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with wave.open(str(target), "rb") as clip:
                fmt = (clip.getnchannels(), clip.getsampwidth(), clip.getframerate())
                if fmt != (1, _SAMPLE_WIDTH, sample_rate):
                    logger.warning("unexpected spoken cue format %s", fmt)
                    return None
                return clip.readframes(clip.getnframes())
    except (OSError, subprocess.SubprocessError, wave.Error):
        logger.warning("could not render the spoken cue", exc_info=True)
        return None


@dataclass
class AudioCuePlayer:
    """Plays a short cue through a stream that is already open."""

    config: AudioConfig
    stream_factory: OutputStreamFactory | None = None
    pcm: bytes = b""
    text: str = "Amy here"
    voice: str | None = None
    is_playing: threading.Event = field(default_factory=threading.Event, init=False)
    _stream: _OutputStream | None = field(default=None, init=False, repr=False)
    _cursor: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._cursor = len(self.pcm)

    @property
    def is_available(self) -> bool:
        """True once an output stream is open, so callers know the cue will be heard."""
        return self._stream is not None

    @property
    def duration_seconds(self) -> float:
        return len(self.pcm) / _SAMPLE_WIDTH / self.config.sample_rate

    def start(self) -> None:
        if self._stream is not None:
            return

        if not self.pcm:
            rendered = render_speech_pcm(
                self.text,
                sample_rate=self.config.sample_rate,
                voice=self.voice,
            )
            if rendered is None:
                logger.warning("spoken wake cue unavailable; continuing without it")
                return
            self.pcm = rendered
            self._cursor = len(self.pcm)

        factory = self.stream_factory or _default_stream_factory
        try:
            stream = factory(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=256,
                callback=self._fill,
            )
            stream.start()
        except Exception:
            # A missing or busy output device must not take the assistant down; the
            # acknowledgement is a nicety and the rest of the pipeline is unaffected.
            logger.warning("wake cue unavailable; continuing without it", exc_info=True)
            return
        self._stream = stream

    def play(self) -> None:
        if self._stream is None:
            return
        with self._lock:
            self._cursor = 0
        self.is_playing.set()

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        self.is_playing.clear()
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            logger.debug("failed to close cue stream", exc_info=True)

    def _fill(self, outdata: _OutputBuffer, frames: int, _time: object, _status: object) -> None:
        wanted = frames * _SAMPLE_WIDTH
        with self._lock:
            chunk = self.pcm[self._cursor : self._cursor + wanted]
            self._cursor += len(chunk)
            drained = self._cursor >= len(self.pcm)

        outdata[0 : len(chunk)] = chunk
        if len(chunk) < wanted:
            outdata[len(chunk) : wanted] = bytes(wanted - len(chunk))
        if drained:
            self.is_playing.clear()


def _default_stream_factory(**kwargs: object) -> _OutputStream:
    import sounddevice as sd

    return cast(_OutputStream, sd.RawOutputStream(**kwargs))  # pyright: ignore[reportUnknownMemberType]


__all__ = ["AudioCuePlayer", "OutputStreamFactory", "render_speech_pcm"]

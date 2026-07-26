from __future__ import annotations

from dataclasses import dataclass, field
import logging
import platform
import subprocess
import threading
from typing import Protocol, cast

logger = logging.getLogger(__name__)


class _TtsEngine(Protocol):
    def setProperty(self, name: str, value: object) -> None: ...

    def say(self, text: str) -> None: ...

    def runAndWait(self) -> None: ...


@dataclass
class LocalSpeaker:
    voice: str | None = None
    _process: subprocess.Popen[str] | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    is_speaking: threading.Event = field(default_factory=threading.Event, init=False)
    _speak_token: int = field(default=0, init=False, repr=False)
    _active_token: int | None = field(default=None, init=False, repr=False)

    def speak(self, text: str) -> None:
        logger.debug("tts speak requested: %r", text)
        self.stop()
        with self._lock:
            self._speak_token += 1
            speak_token = self._speak_token
            self._active_token = speak_token
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
                if self._active_token == speak_token:
                    self._process = None
                    self._active_token = None
                    self.is_speaking.clear()

    def stop(self) -> None:
        logger.debug("tts stop requested")
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=1)
            self._process = None
            self._active_token = None
            self.is_speaking.clear()

__all__ = ["LocalSpeaker"]

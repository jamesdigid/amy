from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import re
import threading
import time
import logging
from typing import Protocol, TypedDict, cast

from .models import AudioConfig
from .vad import VadState
from ...understanding.interpreter import DEFAULT_WAKE_WORD_ALIASES

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class WakeDetection:
    matched: bool
    transcript: str = ""
    trailing_text: str = ""

    @property
    def is_bare(self) -> bool:
        """True when the wake word arrived with nothing after it.

        A spoken acknowledgement lasts long enough to talk over a command, so the
        greeting is reserved for a bare wake word. Anything trailing means the user
        is already mid-request and wants an answer, not a greeting.
        """
        return self.matched and not self.trailing_text


class WakeDetector(Protocol):
    def detect(self, pcm: bytes, *, vad_state: VadState | None = None) -> WakeDetection: ...

    def warmup(self) -> None: ...

    def reset(self) -> None: ...


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


@dataclass
class WhisperWakeDetector:
    config: AudioConfig
    wake_word: str | None = None
    model_repo: str | None = None
    language: str = "en"
    prefixes: tuple[str, ...] = ("hey", "hi")
    _model_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _module: _MlxWhisperModule | None = field(default=None, init=False, repr=False)
    _warmed: bool = field(default=False, init=False, repr=False)
    _cooldown_until: float = field(default=0.0, init=False, repr=False)
    _pattern: re.Pattern[str] | None = field(default=None, init=False, repr=False)

    @property
    def repo(self) -> str:
        return self.model_repo or self.config.wake_model_repo

    def _wake_aliases(self) -> tuple[str, ...]:
        if self.wake_word is None:
            return DEFAULT_WAKE_WORD_ALIASES
        normalized = self.wake_word.lower().strip()
        if normalized == "amy":
            return DEFAULT_WAKE_WORD_ALIASES
        return (normalized,)

    def _load_module(self) -> _MlxWhisperModule:
        if self._module is not None:
            return self._module

        with self._model_lock:
            if self._module is None:
                self._module = cast(_MlxWhisperModule, importlib.import_module("mlx_whisper"))
        return self._module

    def warmup(self) -> None:
        if self._warmed:
            return

        module = self._load_module()
        with self._model_lock:
            if self._warmed:
                return
            import numpy as np

            silence = np.zeros(self.config.sample_rate, dtype=np.float32)
            module.transcribe(
                silence,
                path_or_hf_repo=self.repo,
                language=self.language,
                verbose=None,
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            self._warmed = True

    def reset(self) -> None:
        self._cooldown_until = 0.0

    def detect(self, pcm: bytes, *, vad_state: VadState | None = None) -> WakeDetection:
        if time.monotonic() < self._cooldown_until:
            return WakeDetection(matched=False)
        if vad_state is not None and not vad_state.is_speech:
            return WakeDetection(matched=False)

        import numpy as np

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        audio = samples / 32768.0
        result = self._load_module().transcribe(
            audio,
            path_or_hf_repo=self.repo,
            language=self.language,
            verbose=None,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        text = _normalize_text(result["text"])
        # rms is in the same int16 units the VAD compares, so it reads directly
        # against the gate the adaptive VAD reports when the floor moves.
        logger.debug(
            "wake transcript: %r (%.2fs, rms %d, peak %.3f)",
            text,
            len(audio) / self.config.sample_rate,
            int(np.sqrt(np.square(samples).mean())) if samples.size else 0,
            float(np.abs(audio).max()) if len(audio) else 0.0,
        )
        if not text:
            return WakeDetection(matched=False)

        match = self._matches_wake_word(text)
        if match is None:
            return WakeDetection(matched=False)

        self._cooldown_until = time.monotonic() + self.config.wake_cooldown_ms / 1000
        trailing_text = text[match.end() :].strip()
        logger.debug("wake detected: %r (trailing %r)", text, trailing_text)
        return WakeDetection(matched=True, transcript=text, trailing_text=trailing_text)

    def _matches_wake_word(self, text: str) -> re.Match[str] | None:
        """Match the wake word at the head of the transcript, followed by anything.

        The command usually arrives in the same breath ("amy what is the weather"),
        so requiring the alias to be the whole transcript would only ever match a
        bare "amy". The trailing \\b stops "amybody" from counting.
        """
        if self._pattern is None:
            aliases = "|".join(re.escape(alias) for alias in self._wake_aliases())
            prefix_group = ""
            if self.prefixes:
                prefixes = "|".join(re.escape(prefix) for prefix in self.prefixes)
                prefix_group = rf"(?:(?:{prefixes})\s+)?"
            self._pattern = re.compile(rf"^{prefix_group}(?:{aliases})\b")
        return self._pattern.search(text)


@dataclass
class StubWakeDetector:
    should_detect: bool = False
    trailing_text: str = ""
    detections: int = 0
    polls: int = 0

    def detect(self, pcm: bytes, *, vad_state: VadState | None = None) -> WakeDetection:  # noqa: ARG002
        self.polls += 1
        if self.should_detect:
            self.detections += 1
            self.should_detect = False
            return WakeDetection(
                matched=True,
                transcript=f"amy {self.trailing_text}".strip(),
                trailing_text=self.trailing_text,
            )
        return WakeDetection(matched=False)

    def warmup(self) -> None:
        return None

    def reset(self) -> None:
        self.should_detect = False


from __future__ import annotations

from dataclasses import dataclass, field
from array import array
import importlib
from pathlib import Path
import threading
from typing import Protocol, TypedDict, cast


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
    _module: _MlxWhisperModule | None = field(default=None, init=False, repr=False)
    _warmed: bool = field(default=False, init=False, repr=False)

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
            verbose=None,
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
            silence = array("f", [0.0] * 16000)
            module.transcribe(
                silence,
                path_or_hf_repo=self.model_repo,
                language=self.language,
                verbose=None,
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


__all__ = ["MlxWhisperTranscriber", "StubTranscriber", "Transcriber"]

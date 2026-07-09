from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, cast


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

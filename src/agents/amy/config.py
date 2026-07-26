from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class AppConfig:
    api_key: str
    model: str
    assistant_name: str
    project_context_path: Path
    memory_dir: Path
    recent_turns: int
    wake_word: str
    transcript_language: str | None = None
    transcription_model: str = "mlx-community/whisper-large-v3-turbo"
    wake_model: str = "mlx-community/whisper-tiny"
    ring_buffer_ms: int = 2000
    wake_window_ms: int = 1200
    wake_min_window_ms: int = 500
    wake_poll_ms: int = 300
    rms_threshold: int = 500
    wake_rms_threshold: int = 650
    silence_ms: int = 700
    log_transcripts: bool = False
    audio_input_device: str | int | None = None

    @property
    def project_context(self) -> str:
        if not self.project_context_path.exists():
            return ""
        return self.project_context_path.read_text(encoding="utf-8").strip()


def load_config(base_dir: Path | None = None) -> AppConfig:
    workspace = base_dir or Path.cwd()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")

    model = _get_env_with_fallback("AMY_MODEL", "AIMEE_MODEL", "gpt-4.1-mini")
    assistant_name = _get_env_with_fallback("AMY_ASSISTANT_NAME", "AIMEE_ASSISTANT_NAME", "Amy")
    context_path = Path(
        _get_env_with_fallback(
            "AMY_CONTEXT_PATH",
            "AIMEE_CONTEXT_PATH",
            str(workspace / "config" / "project_context.md"),
        )
    )
    memory_dir = Path(
        _get_env_with_fallback(
            "AMY_MEMORY_DIR",
            "AIMEE_MEMORY_DIR",
            str(workspace / "src" / "agents" / "amy" / "memory"),
        )
    )
    recent_turns_text = _get_env_with_fallback("AMY_RECENT_TURNS", "AIMEE_RECENT_TURNS", "6")
    recent_turns = int(recent_turns_text)
    wake_word = _get_env_with_fallback("AMY_WAKE_WORD", "AIMEE_WAKE_WORD", "amy").lower()
    transcript_language_raw = _get_env_with_fallback(
        "AMY_TRANSCRIPT_LANGUAGE", "AIMEE_TRANSCRIPT_LANGUAGE", ""
    )
    transcript_language = transcript_language_raw.strip() or None
    transcription_model = _get_env_with_fallback(
        "AMY_TRANSCRIPTION_MODEL",
        "AIMEE_TRANSCRIPTION_MODEL",
        "mlx-community/whisper-large-v3-turbo",
    )
    wake_model = _get_env_with_fallback("AMY_WAKE_MODEL", "AIMEE_WAKE_MODEL", "mlx-community/whisper-tiny")
    ring_buffer_ms = int(_get_env_with_fallback("AMY_RING_BUFFER_MS", "AIMEE_RING_BUFFER_MS", "2000"))
    wake_window_ms = int(_get_env_with_fallback("AMY_WAKE_WINDOW_MS", "AIMEE_WAKE_WINDOW_MS", "1200"))
    wake_min_window_ms = int(
        _get_env_with_fallback("AMY_WAKE_MIN_WINDOW_MS", "AIMEE_WAKE_MIN_WINDOW_MS", "500")
    )
    wake_poll_ms = int(_get_env_with_fallback("AMY_WAKE_POLL_MS", "AIMEE_WAKE_POLL_MS", "300"))
    rms_threshold = int(_get_env_with_fallback("AMY_RMS_THRESHOLD", "AIMEE_RMS_THRESHOLD", "500"))
    wake_rms_threshold = int(
        _get_env_with_fallback("AMY_WAKE_RMS_THRESHOLD", "AIMEE_WAKE_RMS_THRESHOLD", "650")
    )
    silence_ms = int(_get_env_with_fallback("AMY_SILENCE_MS", "AIMEE_SILENCE_MS", "700"))
    log_transcripts_raw = _get_env_with_fallback(
        "AMY_LOG_TRANSCRIPTS", "AIMEE_LOG_TRANSCRIPTS", "false"
    )
    log_transcripts = _parse_bool(log_transcripts_raw)
    audio_input_device_raw = _get_env_with_fallback(
        "AMY_AUDIO_INPUT_DEVICE",
        "AIMEE_AUDIO_INPUT_DEVICE",
        "",
    )
    audio_input_device = _parse_optional_device(audio_input_device_raw)

    return AppConfig(
        api_key=api_key,
        model=model,
        assistant_name=assistant_name,
        project_context_path=context_path,
        memory_dir=memory_dir,
        recent_turns=recent_turns,
        wake_word=wake_word,
        transcript_language=transcript_language,
        transcription_model=transcription_model,
        wake_model=wake_model,
        ring_buffer_ms=ring_buffer_ms,
        wake_window_ms=wake_window_ms,
        wake_min_window_ms=wake_min_window_ms,
        wake_poll_ms=wake_poll_ms,
        rms_threshold=rms_threshold,
        wake_rms_threshold=wake_rms_threshold,
        silence_ms=silence_ms,
        log_transcripts=log_transcripts,
        audio_input_device=audio_input_device,
    )


def _get_env_with_fallback(primary: str, legacy: str, default: str) -> str:
    for name in (primary, legacy):
        value = os.environ.get(name)
        if value is None:
            continue

        value = value.strip()
        if value:
            return value
    return default


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_optional_device(value: str) -> str | int | None:
    device = value.strip()
    if not device:
        return None
    try:
        return int(device)
    except ValueError:
        return device

__all__ = ["AppConfig", "load_config"]

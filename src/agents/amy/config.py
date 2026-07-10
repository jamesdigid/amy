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
    log_transcripts: bool = False

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
    log_transcripts_raw = _get_env_with_fallback(
        "AMY_LOG_TRANSCRIPTS", "AIMEE_LOG_TRANSCRIPTS", "false"
    )
    log_transcripts = _parse_bool(log_transcripts_raw)

    return AppConfig(
        api_key=api_key,
        model=model,
        assistant_name=assistant_name,
        project_context_path=context_path,
        memory_dir=memory_dir,
        recent_turns=recent_turns,
        wake_word=wake_word,
        transcript_language=transcript_language,
        log_transcripts=log_transcripts,
    )


def _get_env_with_fallback(primary: str, legacy: str, default: str) -> str:
    value = os.environ.get(primary)
    if value is None:
        value = os.environ.get(legacy, default)
    return value.strip()


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}

__all__ = ["AppConfig", "load_config"]

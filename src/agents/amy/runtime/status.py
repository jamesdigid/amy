from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re

from ..core.models import AssistantStatus
from ..skills.registry import AmySkillRegistry, SkillSmokeResult

MAX_SKILL_STEM_LENGTH = 100
MAX_SKILL_TAGS = 10
logger = logging.getLogger(__name__)

_SKILL_NOTE_TAGS = {
    "capabilities",
    "capability",
    "plan",
    "plans",
    "skill",
    "skills",
    "status",
}


@dataclass(frozen=True)
class AmyStatusReporter:
    memory_dir: Path
    skill_registry: AmySkillRegistry | None = None
    web_search_enabled: bool = True
    transcript_logging_enabled: bool = False
    max_skill_notes: int = 3

    def build_report(self, status: AssistantStatus) -> str:
        logger.debug("building status report")
        runtime_summary = self._build_runtime_summary(status)
        capability_summary = self._build_capability_summary()
        registered_skills = self._build_registered_skills()
        smoke_test_summary = self._build_smoke_test_summary()
        skill_notes = self._build_skill_notes()

        parts = [
            f"Status check: {runtime_summary}.",
            f"Capabilities: {capability_summary}.",
        ]
        if registered_skills:
            parts.append(f"Registered skills: {', '.join(registered_skills)}.")
        if smoke_test_summary:
            parts.append(f"Smoke test: {smoke_test_summary}.")
        if skill_notes:
            parts.append(f"Skill notes: {', '.join(skill_notes)}.")
        return " ".join(parts)

    def _build_runtime_summary(self, status: AssistantStatus) -> str:
        phase = getattr(status, "phase", None)
        phase_value = getattr(phase, "value", "unknown")
        paused = bool(getattr(status, "paused", False))
        active_conversation = bool(getattr(status, "active_conversation", False))
        error_message = str(getattr(status, "error_message", "")).strip()
        parts = [
            phase_value,
            "paused" if paused else "not paused",
            "active conversation" if active_conversation else "no active conversation",
        ]
        if error_message:
            parts.append(f"error: {error_message}")
        else:
            parts.append("no errors")
        return ", ".join(parts)

    def _build_capability_summary(self) -> str:
        parts = [
            "wake-word voice input",
            "listen/respond",
            "pause/resume/cut",
            "memory save/retrieve",
            "local speech recognition",
            "local speech output",
        ]
        if self.web_search_enabled:
            parts.append("web search")
        if self.transcript_logging_enabled:
            parts.append("transcript logging")
        return ", ".join(parts)

    def _build_skill_notes(self) -> list[str]:
        if not self.memory_dir.exists():
            return []

        notes: list[str] = []
        for path in sorted(self.memory_dir.glob("*.md"), key=lambda item: item.name.lower()):
            if path.name == "memory.md":
                continue
            tags = self._derive_tags(path)
            if not tags or not self._is_skill_note(tags):
                continue

            summary = self._extract_summary(path.read_text(encoding="utf-8"))
            if not summary:
                summary = self._humanize_stem(path.stem)
            notes.append(summary)
            if len(notes) >= self.max_skill_notes:
                break
        return notes

    def _build_registered_skills(self) -> list[str]:
        if self.skill_registry is None:
            logger.debug("status skill list skipped: no skill registry")
            return []
        registered_skills = self.skill_registry.registered_skills()
        logger.debug("status registered skills: %s", ", ".join(registered_skills) if registered_skills else "none")
        return registered_skills

    def _build_smoke_test_summary(self) -> str:
        if self.skill_registry is None:
            logger.debug("status smoke test skipped: no skill registry")
            return ""

        logger.debug("status smoke test running")
        results = self.skill_registry.smoke_test()
        if not results:
            logger.debug("status smoke test returned no results")
            return ""

        passed_count = sum(1 for result in results if result.passed)
        detail = ", ".join(self._format_smoke_result(result) for result in results)
        logger.debug("status smoke test summary: %s/%s passed", passed_count, len(results))
        return f"{passed_count}/{len(results)} passed ({detail})"

    def _derive_tags(self, path: Path) -> tuple[str, ...]:
        stem = path.stem.lower()
        if len(stem) > MAX_SKILL_STEM_LENGTH:
            return ()

        tags = tuple(tag for tag in stem.split(".") if tag)
        if not tags or len(tags) > MAX_SKILL_TAGS:
            return ()
        return tags

    def _is_skill_note(self, tags: tuple[str, ...]) -> bool:
        return any(tag in _SKILL_NOTE_TAGS for tag in tags)

    def _extract_summary(self, content: str) -> str:
        lines = content.splitlines()
        for index, line in enumerate(lines):
            if line.strip().lower() != "## summary":
                continue
            for summary_line in lines[index + 1 :]:
                stripped = summary_line.strip()
                if stripped:
                    return stripped.lstrip("- ").strip()
            break
        return ""

    def _humanize_stem(self, stem: str) -> str:
        text = re.sub(r"[._-]+", " ", stem).strip()
        return text[:1].upper() + text[1:] if text else ""

    def _format_smoke_result(self, result: SkillSmokeResult) -> str:
        status_text = "ok" if result.passed else "failed"
        if result.details:
            return f"{result.name} {status_text}: {result.details}"
        return f"{result.name} {status_text}"


__all__ = ["AmyStatusReporter"]

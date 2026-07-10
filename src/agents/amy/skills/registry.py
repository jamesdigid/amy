from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re
import subprocess
import tempfile
import sys

from ..memory import MemoryStore, MemoryStoreProtocol
from .browser import SearchResult, WebSearcher

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SkillSmokeResult:
    name: str
    passed: bool
    details: str = ""


@dataclass
class AmySkillRegistry:
    project_root: Path
    memory_store: MemoryStoreProtocol | None = None
    web_search: WebSearcher | None = None
    smoke_query: str = "amy status smoke test"

    def registered_skills(self) -> list[str]:
        skills: list[str] = []
        if self.memory_store is not None:
            skills.extend(["memory retrieve", "memory draft", "memory save"])
        if self.web_search is not None:
            skills.append("web search")
        return skills

    def smoke_test(self) -> list[SkillSmokeResult]:
        logger.debug("starting skill smoke test")
        results = [
            self._smoke_memory_retrieve(),
            self._smoke_memory_draft(),
            self._smoke_memory_save(),
            self._smoke_web_search(),
            self._smoke_test_suite(),
        ]
        logger.debug("completed skill smoke test: %s", ", ".join(
            f"{result.name}={'ok' if result.passed else 'failed'}" for result in results
        ))
        return results

    def _smoke_memory_retrieve(self) -> SkillSmokeResult:
        if self.memory_store is None:
            logger.debug("smoke memory retrieve skipped: memory store not registered")
            return SkillSmokeResult(name="memory retrieve", passed=False, details="memory store not registered")

        try:
            logger.debug("smoke memory retrieve: querying %r", self.smoke_query)
            context = self.memory_store.retrieve_context(self.smoke_query, limit=1)
        except Exception as exc:  # pragma: no cover - defensive runtime path
            logger.exception("smoke memory retrieve failed")
            return SkillSmokeResult(name="memory retrieve", passed=False, details=str(exc))
        detail = (
            "retrieval returned matching memory context"
            if context.strip()
            else "retrieval completed with no matching memory context"
        )
        logger.debug("smoke memory retrieve passed: %s", detail)
        return SkillSmokeResult(name="memory retrieve", passed=True, details=detail)

    def _smoke_memory_draft(self) -> SkillSmokeResult:
        if self.memory_store is None:
            logger.debug("smoke memory draft skipped: memory store not registered")
            return SkillSmokeResult(name="memory draft", passed=False, details="memory store not registered")

        try:
            logger.debug("smoke memory draft: drafting from prompt")
            draft = self.memory_store.draft_from_prompt("remember that amy status smoke test is healthy")
        except Exception as exc:  # pragma: no cover - defensive runtime path
            logger.exception("smoke memory draft failed")
            return SkillSmokeResult(name="memory draft", passed=False, details=str(exc))
        if draft is None:
            logger.debug("smoke memory draft failed: no draft produced")
            return SkillSmokeResult(name="memory draft", passed=False, details="no draft was produced")
        logger.debug("smoke memory draft passed: %s", draft.path.name)
        return SkillSmokeResult(name="memory draft", passed=True, details=draft.path.name)

    def _smoke_memory_save(self) -> SkillSmokeResult:
        if self.memory_store is None:
            logger.debug("smoke memory save skipped: memory store not registered")
            return SkillSmokeResult(name="memory save", passed=False, details="memory store not registered")

        try:
            logger.debug("smoke memory save: using temporary memory store")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_store = MemoryStore(memory_dir=Path(temp_dir))
                draft = temp_store.draft_from_prompt("remember that amy status smoke test save is healthy")
                if draft is None:
                    logger.debug("smoke memory save failed: no draft produced")
                    return SkillSmokeResult(name="memory save", passed=False, details="no draft was produced")
                saved_path = temp_store.save_draft(draft)
                if not saved_path.exists():
                    logger.debug("smoke memory save failed: file missing after write")
                    return SkillSmokeResult(name="memory save", passed=False, details="saved file was not created")
        except Exception as exc:  # pragma: no cover - defensive runtime path
            logger.exception("smoke memory save failed")
            return SkillSmokeResult(name="memory save", passed=False, details=str(exc))
        logger.debug("smoke memory save passed")
        return SkillSmokeResult(name="memory save", passed=True, details="saved to temp memory")

    def _smoke_web_search(self) -> SkillSmokeResult:
        if self.web_search is None:
            logger.debug("smoke web search skipped: web search not registered")
            return SkillSmokeResult(name="web search", passed=False, details="web search not registered")

        try:
            logger.debug("smoke web search: querying %r", self.smoke_query)
            results: list[SearchResult] = self.web_search.search(self.smoke_query, limit=1)
        except Exception as exc:  # pragma: no cover - defensive runtime path
            logger.exception("smoke web search failed")
            return SkillSmokeResult(name="web search", passed=False, details=str(exc))
        logger.debug("smoke web search passed: %d result(s)", len(results))
        return SkillSmokeResult(name="web search", passed=True, details=f"{len(results)} result(s)")

    def _smoke_test_suite(self) -> SkillSmokeResult:
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
        logger.debug("smoke test suite: running %s in %s", " ".join(command), self.project_root)
        try:
            completed = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime path
            logger.exception("smoke test suite failed to launch")
            return SkillSmokeResult(name="test suite", passed=False, details=str(exc))

        output = "\n".join(
            part for part in (completed.stdout.strip(), completed.stderr.strip()) if part
        )
        if completed.returncode != 0:
            logger.debug("smoke test suite failed with code %s", completed.returncode)
            return SkillSmokeResult(name="test suite", passed=False, details=self._summarize_test_output(output))

        logger.debug("smoke test suite passed")
        return SkillSmokeResult(name="test suite", passed=True, details=self._summarize_test_output(output))

    def _summarize_test_output(self, output: str) -> str:
        if not output:
            return "no output"
        match = re.search(r"Ran (\d+) tests?", output)
        if match is not None:
            return f"Ran {match.group(1)} tests"
        return output.splitlines()[-1].strip()


__all__ = ["AmySkillRegistry", "SkillSmokeResult"]

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.amy.models import AssistantStatus
from agents.amy.runtime.status import AmyStatusReporter
from agents.amy.skills.registry import SkillSmokeResult


class FakeSkillRegistry:
    def registered_skills(self) -> list[str]:
        return [
            "memory retrieve",
            "memory draft",
            "memory save",
            "web search",
        ]

    def smoke_test(self) -> list[SkillSmokeResult]:
        return [
            SkillSmokeResult(name="memory retrieve", passed=True, details="returned 0 chars"),
            SkillSmokeResult(name="memory draft", passed=True, details="favorite.editor.vim.md"),
            SkillSmokeResult(name="memory save", passed=True, details="saved to temp memory"),
            SkillSmokeResult(name="web search", passed=True, details="1 result(s)"),
            SkillSmokeResult(name="test suite", passed=True, details="Ran 29 tests"),
        ]


class StatusReporterTests(unittest.TestCase):
    def test_build_report_includes_skill_notes_and_ignores_unrelated_memories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_dir = Path(temp_dir)
            (memory_dir / "memory.md").write_text("# Template\nignore me", encoding="utf-8")
            (memory_dir / "skill.bundles.markdown.storage.mechanism.md").write_text(
                "# Memory\n\n## Summary\nSkill bundles and skill markdown storage mechanism\n",
                encoding="utf-8",
            )
            (memory_dir / "user.plans.create.status.check.md").write_text(
                "# Memory\n\n## Summary\nUser plans to create a status check for assistant rebuilds\n",
                encoding="utf-8",
            )
            (memory_dir / "james.awesome.md").write_text(
                "# Memory\n\n## Summary\nPersonal memory that should not be treated as a skill\n",
                encoding="utf-8",
            )
            reporter = AmyStatusReporter(
                memory_dir=memory_dir,
                skill_registry=FakeSkillRegistry(),
                web_search_enabled=False,
                transcript_logging_enabled=True,
            )

            report = reporter.build_report(AssistantStatus())

            self.assertIn("Status check: idle", report)
            self.assertIn("Capabilities:", report)
            self.assertIn("Registered skills:", report)
            self.assertIn("test suite", report)
            self.assertIn("Smoke test: 5/5 passed", report)
            self.assertIn("transcript logging", report)
            self.assertIn("Skill bundles and skill markdown storage mechanism", report)
            self.assertIn("User plans to create a status check for assistant rebuilds", report)
            self.assertNotIn("Personal memory that should not be treated as a skill", report)
            self.assertNotIn("memory.md", report)


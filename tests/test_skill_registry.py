from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

from agents.amy.memory import MemoryStore
from agents.amy.skills.browser import SearchResult
from agents.amy.skills.registry import AmySkillRegistry


class FakeWebSearch:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, limit: int = 4) -> list[SearchResult]:
        self.queries.append((query, limit))
        return [SearchResult(title="Result", url="https://example.com", snippet="Snippet")]


class SkillRegistryTests(unittest.TestCase):
    def test_smoke_test_runs_registered_memory_and_web_operations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_store = MemoryStore(memory_dir=Path(temp_dir))
            web_search = FakeWebSearch()
            registry = AmySkillRegistry(project_root=Path(temp_dir), memory_store=memory_store, web_search=web_search)

            completed = type(
                "CompletedProcess",
                (),
                {
                    "returncode": 0,
                    "stdout": "Ran 29 tests in 0.123s\nOK\n",
                    "stderr": "",
                },
            )()
            with patch("agents.amy.skills.registry.subprocess.run", return_value=completed) as mock_run:
                results = registry.smoke_test()

            self.assertEqual([result.name for result in results], [
                "memory retrieve",
                "memory draft",
                "memory save",
                "web search",
                "test suite",
            ])
            self.assertTrue(all(result.passed for result in results))
            self.assertEqual(web_search.queries, [("amy status smoke test", 1)])
            self.assertEqual(
                mock_run.call_args.args[0],
                [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            )



from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.amy.memory import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def test_retrieve_context_matches_dot_delimited_filename_tags_and_orders_by_specificity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_dir = Path(temp_dir)
            (memory_dir / "memory.md").write_text("# Template\nignore me", encoding="utf-8")
            (memory_dir / "james.md").write_text("General James memory.", encoding="utf-8")
            (memory_dir / "james.research.md").write_text(
                "Research memory for James.",
                encoding="utf-8",
            )
            (memory_dir / "client.acme.md").write_text("Client memory.", encoding="utf-8")

            store = MemoryStore(memory_dir=memory_dir)
            context = store.retrieve_context("please remember james research")

            self.assertIn("### Memory: james.research.md", context)
            self.assertIn("Tags: james, research", context)
            self.assertIn("Research memory for James.", context)
            self.assertIn("### Memory: james.md", context)
            self.assertNotIn("memory.md", context)
            self.assertLess(
                context.index("### Memory: james.research.md"),
                context.index("### Memory: james.md"),
            )

    def test_retrieve_context_ignores_files_that_exceed_filename_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_dir = Path(temp_dir)
            valid_name = "alpha.beta.md"
            too_many_tags = "a.b.c.d.e.f.g.h.i.j.k.md"
            too_long_name = f"{'x' * 101}.md"

            (memory_dir / valid_name).write_text("Valid memory.", encoding="utf-8")
            (memory_dir / too_many_tags).write_text("Too many tags.", encoding="utf-8")
            (memory_dir / too_long_name).write_text("Too long.", encoding="utf-8")

            store = MemoryStore(memory_dir=memory_dir)
            context = store.retrieve_context("alpha beta")

            self.assertIn("### Memory: alpha.beta.md", context)
            self.assertNotIn(too_many_tags, context)
            self.assertNotIn(too_long_name, context)

    def test_retrieve_context_returns_empty_when_no_memory_dir_exists(self) -> None:
        store = MemoryStore(memory_dir=Path("/tmp/does-not-exist-for-amy"))

        self.assertEqual(store.retrieve_context("anything at all"), "")

    def test_retrieve_context_ignores_template_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_dir = Path(temp_dir)
            (memory_dir / "memory.md").write_text("# Template\nmemory content", encoding="utf-8")

            store = MemoryStore(memory_dir=memory_dir)

            self.assertEqual(store.retrieve_context("memory"), "")

    def test_draft_from_prompt_builds_expected_filename_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(memory_dir=Path(temp_dir))

            draft = store.draft_from_prompt("um, eh, remember that my favorite editor is vim")

            assert draft is not None
            self.assertEqual(draft.path.name, "favorite.editor.vim.md")
            self.assertIn("## Summary", draft.content)
            self.assertIn("favorite editor is vim", draft.content.lower())
            self.assertNotIn("um", draft.path.name)
            self.assertNotIn("eh", draft.path.name)

    def test_draft_from_prompt_uses_subject_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(memory_dir=Path(temp_dir))

            draft = store.draft_from_prompt("remember this", subject="sky is blue")

            assert draft is not None
            self.assertEqual(draft.path.name, "sky.blue.md")
            self.assertIn("Sky is blue", draft.content)

    def test_save_draft_appends_to_an_existing_memory_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(memory_dir=Path(temp_dir))
            draft = store.draft_from_prompt("remember that my favorite editor is vim")

            assert draft is not None
            first_path = store.save_draft(draft)
            second_path = store.save_draft(draft)

            self.assertEqual(first_path, second_path)
            saved = first_path.read_text(encoding="utf-8")
            self.assertIn("---", saved)
            self.assertGreaterEqual(saved.count("## Summary"), 2)

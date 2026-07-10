from __future__ import annotations

from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from agents.amy.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_reads_context_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            context_path = base_path / "config" / "project_context.md"
            context_path.parent.mkdir(parents=True)
            context_path.write_text("Use short answers.", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-key",
                    "AIMEE_CONTEXT_PATH": str(context_path),
                    "AIMEE_MODEL": "gpt-test",
                    "AMY_ASSISTANT_NAME": "Amy",
                    "AMY_LOG_TRANSCRIPTS": "1",
                },
                clear=False,
            ):
                config = load_config(base_path)
                self.assertEqual(config.api_key, "test-key")
                self.assertEqual(config.model, "gpt-test")
                self.assertEqual(config.project_context, "Use short answers.")
                self.assertEqual(config.memory_dir, base_path / "src" / "agents" / "amy" / "memory")
                self.assertTrue(config.log_transcripts)

    def test_load_config_reads_memory_dir_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            memory_dir = base_path / "notes" / "memories"
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-key",
                    "AMY_MEMORY_DIR": str(memory_dir),
                },
                clear=False,
            ):
                config = load_config(base_path)
                self.assertEqual(config.memory_dir, memory_dir)

    def test_load_config_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                    load_config(base_path)

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
                    "AMY_TRANSCRIPTION_MODEL": "mlx-community/whisper-small",
                    "AMY_WAKE_MODEL": "mlx-community/whisper-tiny",
                    "AMY_RING_BUFFER_MS": "2400",
                    "AMY_WAKE_WINDOW_MS": "900",
                    "AMY_WAKE_MIN_WINDOW_MS": "450",
                    "AMY_WAKE_POLL_MS": "150",
                    "AMY_WAKE_RMS_THRESHOLD": "1200",
                    "AMY_LOG_TRANSCRIPTS": "1",
                    "AMY_AUDIO_INPUT_DEVICE": "Audient iD24",
                },
                clear=False,
            ):
                config = load_config(base_path)
                self.assertEqual(config.api_key, "test-key")
                self.assertEqual(config.model, "gpt-test")
                self.assertEqual(config.transcription_model, "mlx-community/whisper-small")
                self.assertEqual(config.project_context, "Use short answers.")
                self.assertEqual(config.memory_dir, base_path / "src" / "agents" / "amy" / "memory")
                self.assertTrue(config.log_transcripts)
                self.assertEqual(config.wake_model, "mlx-community/whisper-tiny")
                self.assertEqual(config.ring_buffer_ms, 2400)
                self.assertEqual(config.wake_window_ms, 900)
                self.assertEqual(config.wake_min_window_ms, 450)
                self.assertEqual(config.wake_poll_ms, 150)
                self.assertEqual(config.wake_rms_threshold, 1200)
                self.assertEqual(config.audio_input_device, "Audient iD24")

    def test_load_config_reads_memory_dir_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            memory_dir = base_path / "notes" / "memories"
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-key",
                    "AMY_MEMORY_DIR": str(memory_dir),
                    "AMY_AUDIO_INPUT_DEVICE": "3",
                },
                clear=False,
            ):
                config = load_config(base_path)
                self.assertEqual(config.memory_dir, memory_dir)
                self.assertEqual(config.audio_input_device, 3)

    def test_load_config_uses_defaults_for_empty_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-key",
                    "AMY_MODEL": "",
                    "AMY_ASSISTANT_NAME": "",
                    "AMY_TRANSCRIPTION_MODEL": "",
                },
                clear=False,
            ):
                config = load_config(base_path)

            self.assertEqual(config.model, "gpt-4.1-mini")
            self.assertEqual(config.assistant_name, "Amy")
            self.assertEqual(config.transcription_model, "mlx-community/whisper-large-v3-turbo")

    def test_load_config_uses_default_transcription_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-key",
                },
                clear=False,
            ):
                config = load_config(base_path)
                self.assertEqual(config.transcription_model, "mlx-community/whisper-large-v3-turbo")

    def test_load_config_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                    load_config(base_path)

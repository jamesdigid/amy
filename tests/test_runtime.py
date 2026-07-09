from __future__ import annotations

import unittest
from pathlib import Path

from amy.audio import AudioConfig
from amy.runtime import AssistantRuntime
from amy.transcription import StubTranscriber


class DummySpeaker:
    def __init__(self) -> None:
        self.is_speaking = None


class DummyController:
    def __init__(self) -> None:
        self.speaker = DummySpeaker()
        self.processed: list[str] = []
        self.status_messages: list[str] = []

    def is_interrupt_command(self, transcript: str) -> bool:
        return transcript.strip().lower() == "pause"

    def process_transcript(self, transcript: str) -> str | None:
        self.processed.append(transcript)
        return None


class RuntimeTests(unittest.TestCase):
    def test_acknowledgement_sound_uses_packaged_assets_directory(self) -> None:
        controller = DummyController()
        runtime = AssistantRuntime(
            controller=controller,  # type: ignore[arg-type]
            transcriber=StubTranscriber("pause"),
            audio_config=AudioConfig(),
            on_status=controller.status_messages.append,
        )

        self.assertEqual(
            Path(runtime._acknowledgement_sound_path).name,  # type: ignore[attr-defined]
            "Glass.aiff",
        )
        self.assertEqual(
            Path(runtime._acknowledgement_sound_path).parent.name,  # type: ignore[attr-defined]
            "assets",
        )

    def test_handle_command_transcript_only_processes_interrupts(self) -> None:
        controller = DummyController()
        runtime = AssistantRuntime(
            controller=controller,  # type: ignore[arg-type]
            transcriber=StubTranscriber("pause"),
            audio_config=AudioConfig(),
            on_status=controller.status_messages.append,
        )

        runtime.on_status("command listener active")
        runtime.handle_command_transcript("pause")
        runtime.handle_command_transcript("hello there")

        self.assertIn("command listener active", controller.status_messages)
        self.assertEqual(controller.processed, ["pause"])

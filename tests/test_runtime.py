from __future__ import annotations

import threading
import unittest
from pathlib import Path

from agents.amy.core.models import AssistantPhase
from agents.amy.modalities.audio import AudioConfig, StubTranscriber
from agents.amy.runtime import AssistantRuntime


class DummySpeaker:
    def __init__(self) -> None:
        self.is_speaking = None


class DummyController:
    def __init__(self) -> None:
        self.speaker = DummySpeaker()
        self.status = type(
            "Status",
            (),
            {
                "phase": AssistantPhase.LISTENING,
                "paused": False,
            },
        )()
        self.processed: list[str] = []
        self.status_messages: list[str] = []

    def is_interrupt_command(self, transcript: str) -> bool:
        return transcript.strip().lower() in {"pause", "stop"}

    def process_transcript(self, transcript: str) -> str | None:
        self.processed.append(transcript)
        return None

    def should_drop_main_transcript(self) -> bool:
        return self.status.paused or self.status.phase in {
            AssistantPhase.SPEAKING,
            AssistantPhase.COOLDOWN,
        }

    def get_status(self) -> object:
        return self.status


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
        runtime.handle_command_transcript("stop")
        runtime.handle_command_transcript("hello there")

        self.assertIn("command listener active", controller.status_messages)
        self.assertEqual(controller.processed, ["pause", "stop"])

    def test_should_queue_main_transcript_honors_speech_and_cooldown_gates(self) -> None:
        controller = DummyController()
        runtime = AssistantRuntime(
            controller=controller,  # type: ignore[arg-type]
            transcriber=StubTranscriber("hello"),
            audio_config=AudioConfig(),
            on_status=controller.status_messages.append,
        )

        self.assertTrue(runtime._should_queue_main_transcript())

        controller.speaker.is_speaking = threading.Event()
        controller.speaker.is_speaking.set()
        self.assertFalse(runtime._should_queue_main_transcript())

        controller.speaker.is_speaking.clear()
        controller.status.phase = AssistantPhase.COOLDOWN
        self.assertFalse(runtime._should_queue_main_transcript())

        controller.status.phase = AssistantPhase.AWAITING_USER_RESPONSE
        self.assertTrue(runtime._should_queue_main_transcript())

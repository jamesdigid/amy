from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from agents.amy.models import AssistantPhase
from agents.amy.modalities.audio import AudioConfig, StubTranscriber
from agents.amy.runtime.assistant import AssistantRuntime
from agents.amy.runtime.status import AmyStatusReporter
import agents.amy.runtime.assistant as runtime_module


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

    def test_status_text_uses_status_reporter(self) -> None:
        controller = DummyController()
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = AmyStatusReporter(memory_dir=Path(temp_dir), web_search_enabled=False)
            runtime = AssistantRuntime(
                controller=controller,  # type: ignore[arg-type]
                transcriber=StubTranscriber("pause"),
                audio_config=AudioConfig(),
                status_reporter=reporter,
                on_status=controller.status_messages.append,
            )

            status_text = runtime.status_text()

            self.assertIn("Status check:", status_text)
            self.assertIn("Capabilities:", status_text)

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

    def test_capture_loop_resets_segmenter_while_main_audio_is_dropped(self) -> None:
        class FakeSegment:
            def __init__(self, path: Path) -> None:
                self.path = path
                self.duration_seconds = 1.0

        class FakeSpeechSegmenter:
            instances: list["FakeSpeechSegmenter"] = []

            def __init__(self, _config: AudioConfig) -> None:
                self.kind = "main" if not FakeSpeechSegmenter.instances else "command"
                self.feed_calls = 0
                self.reset_calls = 0
                self.absolute_feed_count = 0
                self.segment_start_frame = 0
                FakeSpeechSegmenter.instances.append(self)

            def feed(self, _frame: bytes) -> FakeSegment | None:
                self.feed_calls += 1
                self.absolute_feed_count += 1
                if self.feed_calls == 1:
                    self.segment_start_frame = self.absolute_feed_count
                if self.kind == "command":
                    return None
                if self.feed_calls < 3:
                    return None
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_path = Path(temp_file.name)
                temp_file.write(b"RIFF")
                temp_file.close()
                return FakeSegment(temp_path)

            def reset(self) -> None:
                self.reset_calls += 1
                self.feed_calls = 0
                self.segment_start_frame = 0
                return None

        class FakeMicrophoneSource:
            def __init__(self, _config: AudioConfig) -> None:
                self._frames = [b"1", b"2", b"3", b"4", b"5"]

            def __enter__(self) -> "FakeMicrophoneSource":
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

            def frames(self) -> list[bytes]:
                return self._frames

        class FakeTranscriber:
            def __init__(self) -> None:
                self.paths: list[Path] = []

            def transcribe(self, audio_path: Path) -> str:
                self.paths.append(audio_path)
                return "hello"

        class DropThenListenController(DummyController):
            def __init__(self) -> None:
                super().__init__()
                self.drop_calls = 0

            def should_drop_main_transcript(self) -> bool:
                self.drop_calls += 1
                return True

        original_speech_segmenter = runtime_module.SpeechSegmenter
        original_microphone_source = runtime_module.MicrophoneSource
        try:
            runtime_module.SpeechSegmenter = FakeSpeechSegmenter  # type: ignore[assignment]
            runtime_module.MicrophoneSource = FakeMicrophoneSource  # type: ignore[assignment]

            controller = DropThenListenController()
            transcriber = FakeTranscriber()
            runtime = AssistantRuntime(
                controller=controller,  # type: ignore[arg-type]
                transcriber=transcriber,  # type: ignore[arg-type]
                audio_config=AudioConfig(),
                on_status=controller.status_messages.append,
            )
            runtime._capture_enabled.set()

            runtime._capture_loop()

            main_segmenter = FakeSpeechSegmenter.instances[0]
            self.assertEqual(controller.drop_calls, 5)
            self.assertEqual(main_segmenter.reset_calls, 5)
            self.assertEqual(main_segmenter.feed_calls, 0)
        finally:
            runtime_module.SpeechSegmenter = original_speech_segmenter  # type: ignore[assignment]
            runtime_module.MicrophoneSource = original_microphone_source  # type: ignore[assignment]

    def test_capture_loop_cleans_up_transcribed_audio_segments(self) -> None:
        class FakeSegment:
            def __init__(self, path: Path) -> None:
                self.path = path
                self.duration_seconds = 1.0

        class FakeSpeechSegmenter:
            def __init__(self, _config: AudioConfig) -> None:
                self.feed_calls = 0

            def feed(self, _frame: bytes) -> FakeSegment | None:
                self.feed_calls += 1
                if self.feed_calls < 3:
                    return None
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_path = Path(temp_file.name)
                temp_file.write(b"RIFF")
                temp_file.close()
                return FakeSegment(temp_path)

            def reset(self) -> None:
                return None

        class FakeMicrophoneSource:
            def __init__(self, _config: AudioConfig) -> None:
                self._frames = [b"1", b"2", b"3"]

            def __enter__(self) -> "FakeMicrophoneSource":
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

            def frames(self) -> list[bytes]:
                return self._frames

        class FakeTranscriber:
            def __init__(self) -> None:
                self.paths: list[Path] = []

            def transcribe(self, audio_path: Path) -> str:
                self.paths.append(audio_path)
                return "hello"

        original_speech_segmenter = runtime_module.SpeechSegmenter
        original_microphone_source = runtime_module.MicrophoneSource
        try:
            runtime_module.SpeechSegmenter = FakeSpeechSegmenter  # type: ignore[assignment]
            runtime_module.MicrophoneSource = FakeMicrophoneSource  # type: ignore[assignment]

            controller = DummyController()
            transcriber = FakeTranscriber()
            runtime = AssistantRuntime(
                controller=controller,  # type: ignore[arg-type]
                transcriber=transcriber,  # type: ignore[arg-type]
                audio_config=AudioConfig(),
                on_status=controller.status_messages.append,
            )
            runtime._capture_enabled.set()

            runtime._capture_loop()

            self.assertGreaterEqual(len(transcriber.paths), 1)
            self.assertTrue(all(not path.exists() for path in transcriber.paths))
        finally:
            runtime_module.SpeechSegmenter = original_speech_segmenter  # type: ignore[assignment]
            runtime_module.MicrophoneSource = original_microphone_source  # type: ignore[assignment]

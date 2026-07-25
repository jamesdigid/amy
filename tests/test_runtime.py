from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
import time

from agents.amy.models import AssistantPhase, AssistantStatus
from agents.amy.modalities.audio import AudioConfig, AudioFrame, StubTranscriber
from agents.amy.runtime.assistant import AssistantRuntime
from agents.amy.runtime.status import AmyStatusReporter
import agents.amy.runtime.assistant as runtime_module


class DummySpeaker:
    def __init__(self) -> None:
        self.is_speaking: threading.Event | None = None


class DummyController:
    def __init__(self) -> None:
        self.speaker = DummySpeaker()
        self.status = AssistantStatus(phase=AssistantPhase.LISTENING)
        self.processed: list[str] = []
        self.processed_utterance_ids: list[str | None] = []
        self.processed_sources: list[Path | None] = []
        self.status_messages: list[str] = []

    def process_transcript(
        self,
        transcript: str,
        *,
        utterance_id: str | None = None,
        source_path: Path | None = None,
    ) -> str | None:
        self.processed.append(transcript)
        self.processed_utterance_ids.append(utterance_id)
        self.processed_sources.append(source_path)
        return None

    def should_drop_main_transcript(self) -> bool:
        return self.status.paused or self.status.phase in {
            AssistantPhase.SPEAKING,
            AssistantPhase.COOLDOWN,
        }

    def get_status(self) -> AssistantStatus:
        return self.status

    @property
    def epoch(self) -> int:
        return 0


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
                self._frames = [
                    AudioFrame(data=b"1"),
                    AudioFrame(data=b"2"),
                    AudioFrame(data=b"3"),
                    AudioFrame(data=b"4"),
                    AudioFrame(data=b"5"),
                ]

            def __enter__(self) -> "FakeMicrophoneSource":
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

            def frames(self) -> list[AudioFrame]:
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

    def test_capture_loop_enqueues_utterance_metadata_without_transcribing(self) -> None:
        class FakeSegment:
            def __init__(self, pcm: bytes, duration_seconds: float) -> None:
                self.pcm = pcm
                self.duration_seconds = duration_seconds

        class FakeSpeechSegmenter:
            def __init__(self, _config: AudioConfig) -> None:
                self.feed_calls = 0

            def feed(self, _frame: bytes) -> FakeSegment | None:
                self.feed_calls += 1
                if self.feed_calls < 3:
                    return None
                return FakeSegment(b"hello-bytes", 0.09)

            def reset(self) -> None:
                return None

        class FakeMicrophoneSource:
            def __init__(self, _config: AudioConfig) -> None:
                self._frames = [AudioFrame(data=b"1"), AudioFrame(data=b"2"), AudioFrame(data=b"3")]

            def __enter__(self) -> "FakeMicrophoneSource":
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

            def frames(self) -> list[AudioFrame]:
                return self._frames

        class FailingTranscriber:
            def transcribe(self, _audio_path: Path) -> str:
                raise AssertionError("capture loop should not transcribe inline")

        original_speech_segmenter = runtime_module.SpeechSegmenter
        original_microphone_source = runtime_module.MicrophoneSource
        try:
            runtime_module.SpeechSegmenter = FakeSpeechSegmenter  # type: ignore[assignment]
            runtime_module.MicrophoneSource = FakeMicrophoneSource  # type: ignore[assignment]

            controller = DummyController()
            runtime = AssistantRuntime(
                controller=controller,  # type: ignore[arg-type]
                transcriber=FailingTranscriber(),  # type: ignore[arg-type]
                audio_config=AudioConfig(),
                on_status=controller.status_messages.append,
            )
            runtime._capture_enabled.set()

            runtime._capture_loop()

            self.assertEqual(runtime._utterance_queue.qsize(), 1)
            job = runtime._utterance_queue.get_nowait()
            self.assertEqual(job.utterance_id, "utt-1")
            self.assertEqual(job.epoch, 0)
            self.assertEqual(job.pcm, b"hello-bytes")
            self.assertAlmostEqual(job.duration_seconds, 0.09, places=2)
        finally:
            runtime_module.SpeechSegmenter = original_speech_segmenter  # type: ignore[assignment]
            runtime_module.MicrophoneSource = original_microphone_source  # type: ignore[assignment]

    def test_stt_worker_transcribes_utterance_and_cleans_temp_wav(self) -> None:
        stop_event = threading.Event()

        class FakeTranscriber:
            def __init__(self) -> None:
                self.paths: list[Path] = []

            def transcribe(self, audio_path: Path) -> str:
                self.paths.append(audio_path)
                stop_event.set()
                return "hello"

        controller = DummyController()
        transcriber = FakeTranscriber()
        runtime = AssistantRuntime(
            controller=controller,  # type: ignore[arg-type]
            transcriber=transcriber,  # type: ignore[arg-type]
            audio_config=AudioConfig(),
            on_status=controller.status_messages.append,
        )
        runtime._stop_event = stop_event
        runtime._utterance_queue.put(
            runtime_module.UtteranceJob(
                utterance_id="utt-1",
                epoch=0,
                captured_at=time.monotonic(),
                pcm=b"\x01\x02" * 16,
                duration_seconds=0.032,
            )
        )

        worker = threading.Thread(target=runtime._stt_worker_loop)
        worker.start()
        worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(transcriber.paths), 1)
        self.assertFalse(transcriber.paths[0].exists())
        transcript_job = runtime._transcript_queue.get_nowait()
        self.assertEqual(transcript_job.utterance_id, "utt-1")
        self.assertEqual(transcript_job.epoch, 0)
        self.assertEqual(transcript_job.transcript, "hello")

    def test_worker_loop_passes_utterance_id_to_controller(self) -> None:
        stop_event = threading.Event()

        class StopAfterFirstController(DummyController):
            def process_transcript(
                self,
                transcript: str,
                *,
                utterance_id: str | None = None,
                source_path: Path | None = None,
            ) -> str | None:
                result = super().process_transcript(
                    transcript,
                    utterance_id=utterance_id,
                    source_path=source_path,
                )
                stop_event.set()
                return result

        controller = StopAfterFirstController()
        runtime = AssistantRuntime(
            controller=controller,  # type: ignore[arg-type]
            transcriber=StubTranscriber("hello"),
            audio_config=AudioConfig(),
            on_status=controller.status_messages.append,
        )
        runtime._stop_event = stop_event
        runtime._transcript_queue.put(
            runtime_module.TranscriptJob(
                utterance_id="utt-42",
                epoch=0,
                captured_at=time.monotonic(),
                transcript="hello",
            )
        )

        worker = threading.Thread(target=runtime._worker_loop)
        worker.start()
        worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(controller.processed_utterance_ids, ["utt-42"])

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from typing import Callable
import time

from agents.amy.models import AssistantPhase, AssistantStatus
from agents.amy.modalities.audio import (
    AudioConfig,
    AudioFrame,
    AudioRingBuffer,
    CommandRecorder,
    EnergyVad,
    StubTranscriber,
    StubWakeDetector,
)
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
        wake_confirmed: bool = False,
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

    def stop(self) -> None:
        return None


def _silent_frame(samples: int) -> bytes:
    return (b"\x00\x00") * samples


def _loud_frame(samples: int) -> bytes:
    return (b"\xff\x7f") * samples


def _frame_at_rms(samples: int, rms: int) -> bytes:
    return int(rms).to_bytes(2, "little", signed=True) * samples


class FakeSubscription:
    """Replays a fixed frame list, then behaves like a drained subscription."""

    def __init__(self, frames: list[AudioFrame], close_when_drained: bool = True) -> None:
        self._frames = frames
        self._index = 0
        self._close_when_drained = close_when_drained

    def receive(self, timeout: float | None = None) -> AudioFrame | None:  # noqa: ARG002
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        self._index += 1
        return frame

    @property
    def is_closed(self) -> bool:
        return self._close_when_drained and self._index >= len(self._frames)


def _wake_config(**overrides: int) -> AudioConfig:
    defaults: dict[str, int] = {
        "frame_ms": 30,
        "rms_threshold": 100,
        "wake_poll_ms": 90,
        "wake_min_speech_ms": 90,
        "wake_hold_ms": 90,
        "wake_min_window_ms": 30,
    }
    defaults.update(overrides)
    return AudioConfig(**defaults)  # type: ignore[arg-type]


def _speech_run(config: AudioConfig, count: int) -> list[AudioFrame]:
    return [
        AudioFrame(data=_loud_frame(config.frame_samples), sequence=offset + 1)
        for offset in range(count)
    ]


class FakeCuePlayer:
    """Stands in for the output-stream backed cue without touching an audio device."""

    def __init__(self) -> None:
        self.plays = 0
        self.started = 0
        self.closed = 0
        self.is_playing = threading.Event()

    def start(self) -> None:
        self.started += 1

    def play(self) -> None:
        self.plays += 1
        self.is_playing.set()

    def close(self) -> None:
        self.closed += 1
        self.is_playing.clear()


def _build_pipeline_runtime(controller: DummyController, config: AudioConfig) -> AssistantRuntime:
    runtime = AssistantRuntime(
        controller=controller,  # type: ignore[arg-type]
        transcriber=StubTranscriber("hello"),
        audio_config=config,
        cue_player_factory=None,
        on_status=controller.status_messages.append,
    )
    runtime._ring_buffer = AudioRingBuffer(config)
    runtime._vad = EnergyVad(config)
    runtime._wake_vad = EnergyVad(config)
    runtime._recorder = CommandRecorder(config)
    runtime._wake_detector = StubWakeDetector()
    runtime._capture_enabled.set()
    return runtime


def _run_loop(target: object, test: unittest.TestCase) -> None:
    worker = threading.Thread(target=target)  # type: ignore[arg-type]
    worker.start()
    worker.join(timeout=2)
    test.assertFalse(worker.is_alive(), "loop did not exit")


class FakeAudioStream:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.observers: list[object] = []
        self.subscriptions: list[object] = []

    def add_observer(self, observer: object) -> None:
        self.observers.append(observer)

    def subscribe(self, _name: str, maxsize: int = 8) -> FakeSubscription:  # noqa: ARG002
        subscription = FakeSubscription([], close_when_drained=True)
        self.subscriptions.append(subscription)
        return subscription

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


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

    def test_start_stops_before_audio_stream_launch_if_requested(self) -> None:
        warmup_started = threading.Event()
        warmup_release = threading.Event()

        class BlockingTranscriber:
            def warmup(self) -> None:
                warmup_started.set()
                warmup_release.wait(timeout=1)

            def transcribe(self, audio_path: Path) -> str:
                return "hello"

        class BlockingWakeDetector:
            def warmup(self) -> None:
                return None

            def detect(self, pcm: bytes, *, vad_state: object | None = None) -> bool:  # noqa: ARG002
                return False

            def reset(self) -> None:
                return None

        controller = DummyController()
        fake_stream = FakeAudioStream()
        runtime = AssistantRuntime(
            controller=controller,  # type: ignore[arg-type]
            transcriber=BlockingTranscriber(),  # type: ignore[arg-type]
            audio_config=AudioConfig(),
            audio_stream_factory=lambda _config: fake_stream,
            wake_detector_factory=lambda _config: BlockingWakeDetector(),  # type: ignore[arg-type]
            on_status=controller.status_messages.append,
        )

        starter = threading.Thread(target=runtime.start)
        starter.start()
        self.assertTrue(warmup_started.wait(timeout=1))

        runtime.stop()
        warmup_release.set()
        starter.join(timeout=2)

        self.assertFalse(starter.is_alive())
        self.assertEqual(fake_stream.start_calls, 0)

    def test_suspend_keeps_ring_buffer_unless_speaker_is_live(self) -> None:
        config = AudioConfig(frame_ms=30, silence_ms=60, rms_threshold=100)
        controller = DummyController()
        runtime = AssistantRuntime(
            controller=controller,  # type: ignore[arg-type]
            transcriber=StubTranscriber("hello"),
            audio_config=config,
            on_status=controller.status_messages.append,
        )
        runtime._ring_buffer = AudioRingBuffer(config)
        runtime._vad = EnergyVad(config)
        runtime._wake_vad = EnergyVad(config)
        runtime._recorder = CommandRecorder(config)
        runtime._wake_detector = StubWakeDetector()
        runtime._ring_buffer.append(AudioFrame(data=_loud_frame(config.frame_samples), sequence=1))

        # Thinking or cooling down: the pre-roll survives so a fast follow-up is not clipped.
        runtime._suspend_capture()
        self.assertEqual(len(runtime._ring_buffer.snapshot().frames), 1)

        # Speaker live: the only thing on the wire is Amy's own voice, so drop it.
        controller.speaker.is_speaking = threading.Event()
        controller.speaker.is_speaking.set()
        runtime._suspend_capture()
        self.assertEqual(runtime._ring_buffer.snapshot().frames, ())

    def test_pipeline_loop_enqueues_wake_prefixed_utterance(self) -> None:
        config = AudioConfig(frame_ms=30, silence_ms=60, rms_threshold=100)
        controller = DummyController()
        runtime = _build_pipeline_runtime(controller, config)

        prefix = [
            AudioFrame(data=_silent_frame(config.frame_samples), sequence=1),
            AudioFrame(data=_silent_frame(config.frame_samples), sequence=2),
        ]
        for frame in prefix:
            runtime._ring_buffer.append(frame)

        runtime._pending_wake_confirmed = True
        runtime._pending_wake_snapshot = runtime._ring_buffer.snapshot()
        runtime._pipeline_subscription = FakeSubscription(
            [
                AudioFrame(data=_loud_frame(config.frame_samples), sequence=3),
                AudioFrame(data=_loud_frame(config.frame_samples), sequence=4),
                AudioFrame(data=_silent_frame(config.frame_samples), sequence=5),
                AudioFrame(data=_silent_frame(config.frame_samples), sequence=6),
            ]
        )

        _run_loop(runtime._pipeline_loop, self)

        self.assertEqual(runtime._utterance_queue.qsize(), 1)
        job = runtime._utterance_queue.get_nowait()
        self.assertTrue(job.wake_confirmed)
        self.assertTrue(job.pcm.startswith(b"".join(frame.data for frame in prefix)))

    def test_pipeline_loop_drops_audio_while_amy_is_speaking(self) -> None:
        config = AudioConfig(frame_ms=30, silence_ms=60, rms_threshold=100)
        controller = DummyController()
        controller.status.phase = AssistantPhase.SPEAKING
        controller.speaker.is_speaking = threading.Event()
        controller.speaker.is_speaking.set()
        runtime = _build_pipeline_runtime(controller, config)
        runtime._ring_buffer.append(AudioFrame(data=_loud_frame(config.frame_samples), sequence=1))

        runtime._pipeline_subscription = FakeSubscription(
            [
                AudioFrame(data=_loud_frame(config.frame_samples), sequence=2),
                AudioFrame(data=_loud_frame(config.frame_samples), sequence=3),
            ]
        )

        _run_loop(runtime._pipeline_loop, self)

        self.assertEqual(runtime._utterance_queue.qsize(), 0)
        self.assertEqual(runtime._ring_buffer.snapshot().frames, ())

    def test_frame_queue_never_outgrows_the_ring_buffer(self) -> None:
        """A queue deeper than the ring would let a late consumer splice a gap."""
        controller = DummyController()

        def maxsize_for(ring_buffer_ms: int) -> int:
            runtime = AssistantRuntime(
                controller=controller,  # type: ignore[arg-type]
                transcriber=StubTranscriber("hello"),
                audio_config=AudioConfig(frame_ms=30, ring_buffer_ms=ring_buffer_ms),
                on_status=controller.status_messages.append,
            )
            return runtime._frame_queue_maxsize()

        # Short ring: the queue shrinks to match it.
        self.assertLessEqual(maxsize_for(300), AudioConfig(frame_ms=30, ring_buffer_ms=300).ring_buffer_frames)
        self.assertEqual(maxsize_for(300), 10)
        # Generous ring: the queue stays at its own ceiling.
        self.assertEqual(maxsize_for(4000), 32)

    def test_pipeline_loop_drops_partial_recording_on_frame_gap(self) -> None:
        config = AudioConfig(frame_ms=30, silence_ms=60, rms_threshold=100)
        controller = DummyController()
        controller.status.active_conversation = True
        runtime = _build_pipeline_runtime(controller, config)

        # Speech starts a recording, then sequence 9 shows the fan-out lost frames.
        runtime._pipeline_subscription = FakeSubscription(
            [
                AudioFrame(data=_loud_frame(config.frame_samples), sequence=1),
                AudioFrame(data=_loud_frame(config.frame_samples), sequence=2),
                AudioFrame(data=_loud_frame(config.frame_samples), sequence=9),
            ]
        )

        _run_loop(runtime._pipeline_loop, self)

        self.assertEqual(runtime._capture_gap_count, 1)
        self.assertEqual(runtime._utterance_queue.qsize(), 0)

    def test_capture_reset_keeps_confirmed_wake_but_detection_reset_clears_it(self) -> None:
        """A fan-out gap invalidates one consumer's timeline, not the gap-free ring."""
        config = AudioConfig(frame_ms=30, silence_ms=60, rms_threshold=100)
        controller = DummyController()
        runtime = _build_pipeline_runtime(controller, config)
        runtime._recorder.begin(runtime._ring_buffer.snapshot())
        runtime._pending_wake_confirmed = True

        runtime._reset_capture_state()

        self.assertFalse(runtime._recorder.is_recording)
        self.assertTrue(runtime._pending_wake_confirmed)

        runtime._reset_detection_state()

        self.assertFalse(runtime._pending_wake_confirmed)

    def _greet_scenario(
        self,
        *,
        trailing_text: str = "",
        active_conversation: bool = False,
    ) -> tuple[FakeCuePlayer, AssistantRuntime, DummyController]:
        config = _wake_config()
        controller = DummyController()
        controller.status.active_conversation = active_conversation
        runtime = _build_pipeline_runtime(controller, config)
        cue = FakeCuePlayer()
        runtime._cue_player = cue  # type: ignore[assignment]
        detector = StubWakeDetector()
        detector.should_detect = True
        detector.trailing_text = trailing_text
        runtime._wake_detector = detector

        frames = _speech_run(config, count=6)
        for frame in frames:
            runtime._ring_buffer.append(frame)
        runtime._wake_subscription = FakeSubscription(frames)
        _run_loop(runtime._wake_loop, self)
        return cue, runtime, controller

    def test_bare_wake_greets_without_waiting_on_transcription(self) -> None:
        """The wake word is already confirmed, so the greeting must not wait on the recorder."""
        cue, runtime, controller = self._greet_scenario()

        self.assertEqual(cue.plays, 1)
        # Nothing has been segmented or transcribed at this point.
        self.assertEqual(runtime._utterance_queue.qsize(), 0)
        self.assertEqual(controller.processed, [])

    def test_wake_word_followed_by_a_command_does_not_greet(self) -> None:
        """The greeting lasts ~600ms, so it must not talk over a request."""
        cue, runtime, _ = self._greet_scenario(trailing_text="what is the weather")

        self.assertEqual(cue.plays, 0)
        # The command still reaches the pipeline as a confirmed wake.
        self.assertTrue(runtime._pending_wake_confirmed)

    def test_wake_word_mid_conversation_does_not_greet_again(self) -> None:
        cue, _, _ = self._greet_scenario(active_conversation=True)

        self.assertEqual(cue.plays, 0)

    def test_greeting_rearms_once_the_conversation_goes_idle(self) -> None:
        config = _wake_config()
        controller = DummyController()
        runtime = _build_pipeline_runtime(controller, config)
        cue = FakeCuePlayer()
        runtime._cue_player = cue  # type: ignore[assignment]

        runtime._greet_new_conversation()
        self.assertEqual(cue.plays, 1)

        # Live conversation: suppressed.
        controller.status.active_conversation = True
        runtime._greet_new_conversation()
        self.assertEqual(cue.plays, 1)

        # Idle again: the next wake word is greeted like a fresh conversation.
        controller.status.active_conversation = False
        runtime._greet_new_conversation()
        self.assertEqual(cue.plays, 2)

    def test_pipeline_loop_excises_frames_while_the_cue_is_audible(self) -> None:
        config = AudioConfig(frame_ms=30, silence_ms=60, rms_threshold=100)
        controller = DummyController()
        controller.status.active_conversation = True
        runtime = _build_pipeline_runtime(controller, config)
        cue = FakeCuePlayer()
        cue.is_playing.set()
        runtime._cue_player = cue  # type: ignore[assignment]

        runtime._pipeline_subscription = FakeSubscription(
            [
                AudioFrame(data=_loud_frame(config.frame_samples), sequence=1),
                AudioFrame(data=_loud_frame(config.frame_samples), sequence=2),
            ]
        )

        _run_loop(runtime._pipeline_loop, self)

        # Amy's own cue never reaches the recorder.
        self.assertFalse(runtime._recorder.is_recording)
        self.assertEqual(runtime._utterance_queue.qsize(), 0)

    def test_pipeline_loop_keeps_pending_wake_frames_while_cue_is_audible(self) -> None:
        config = AudioConfig(frame_ms=30, silence_ms=60, rms_threshold=100)
        controller = DummyController()
        runtime = _build_pipeline_runtime(controller, config)
        cue = FakeCuePlayer()
        cue.is_playing.set()
        runtime._cue_player = cue  # type: ignore[assignment]
        preroll = AudioFrame(data=_loud_frame(config.frame_samples), sequence=1)
        runtime._ring_buffer.append(preroll)
        runtime._pending_wake_snapshot = runtime._ring_buffer.snapshot()
        runtime._pending_wake_confirmed = True

        runtime._pipeline_subscription = FakeSubscription(
            [AudioFrame(data=_loud_frame(config.frame_samples), sequence=2)]
        )

        _run_loop(runtime._pipeline_loop, self)

        self.assertTrue(runtime._recorder.is_recording)
        self.assertFalse(runtime._pending_wake_confirmed)
        self.assertEqual(runtime._utterance_queue.qsize(), 0)

    def test_wake_loop_restarts_gating_after_frame_gap(self) -> None:
        """Gating counters describe a continuous stream, so a gap has to restart them."""
        config = _wake_config()

        def loud(sequence: int) -> AudioFrame:
            return AudioFrame(data=_loud_frame(config.frame_samples), sequence=sequence)

        def detections_for(frames: list[AudioFrame]) -> tuple[int, int]:
            controller = DummyController()
            runtime = _build_pipeline_runtime(controller, config)
            detector = StubWakeDetector()
            detector.should_detect = True
            runtime._wake_detector = detector
            for frame in frames:
                runtime._ring_buffer.append(frame)
            runtime._wake_subscription = FakeSubscription(frames)
            _run_loop(runtime._wake_loop, self)
            return detector.detections, runtime._wake_gap_count

        # Same audio and frame count either way; only continuity differs.
        self.assertEqual(detections_for([loud(1), loud(2), loud(3), loud(4)]), (1, 0))
        self.assertEqual(detections_for([loud(1), loud(2), loud(20), loud(21)]), (0, 1))

    def test_pipeline_loop_exits_when_audio_stream_dies(self) -> None:
        config = AudioConfig(frame_ms=30, silence_ms=60, rms_threshold=100)
        controller = DummyController()
        runtime = _build_pipeline_runtime(controller, config)
        # No frames and no stop event: only the closed stream can end the loop.
        runtime._pipeline_subscription = FakeSubscription([], close_when_drained=True)

        _run_loop(runtime._pipeline_loop, self)

        self.assertFalse(runtime._stop_event.is_set())

    def test_wake_loop_publishes_detection_with_ring_prefix(self) -> None:
        config = _wake_config()
        controller = DummyController()
        runtime = _build_pipeline_runtime(controller, config)
        detector = StubWakeDetector()
        detector.should_detect = True
        runtime._wake_detector = detector

        frames = _speech_run(config, count=6)
        for frame in frames:
            runtime._ring_buffer.append(frame)
        runtime._wake_subscription = FakeSubscription(frames)

        _run_loop(runtime._wake_loop, self)

        self.assertEqual(detector.detections, 1)
        self.assertTrue(runtime._pending_wake_confirmed)
        self.assertIsNotNone(runtime._pending_wake_snapshot)

    def test_wake_loop_still_polls_after_speech_stops(self) -> None:
        """The window containing the finished wake phrase only ends during trailing silence.

        Gating polls on the current frame being speech means every window ends
        mid-syllable, so the completed phrase is never evaluated.
        """
        config = _wake_config()

        def polls_for(frames: list[AudioFrame]) -> int:
            controller = DummyController()
            runtime = _build_pipeline_runtime(controller, config)
            detector = StubWakeDetector()
            runtime._wake_detector = detector
            for frame in frames:
                runtime._ring_buffer.append(frame)
            runtime._wake_subscription = FakeSubscription(frames)
            _run_loop(runtime._wake_loop, self)
            return detector.polls

        speech = _speech_run(config, count=6)
        trailing_silence = [
            AudioFrame(data=_silent_frame(config.frame_samples), sequence=len(speech) + offset + 1)
            for offset in range(config.wake_hold_frames + 1)
        ]

        self.assertEqual(polls_for(speech), 2)
        self.assertEqual(polls_for(speech + trailing_silence), 3)

    def test_wake_loop_retries_active_speech_on_poll_hop(self) -> None:
        config = _wake_config(wake_poll_ms=90)
        controller = DummyController()
        runtime = _build_pipeline_runtime(controller, config)
        detector = StubWakeDetector()
        runtime._wake_detector = detector
        frames = _speech_run(config, count=12)
        for frame in frames:
            runtime._ring_buffer.append(frame)
        runtime._wake_subscription = FakeSubscription(frames)

        _run_loop(runtime._wake_loop, self)

        self.assertEqual(detector.polls, 4)

    def test_wake_loop_ignores_isolated_noise(self) -> None:
        config = _wake_config()
        controller = DummyController()
        runtime = _build_pipeline_runtime(controller, config)
        detector = StubWakeDetector()
        detector.should_detect = True
        runtime._wake_detector = detector

        frames = _speech_run(config, count=2)
        for frame in frames:
            runtime._ring_buffer.append(frame)
        runtime._wake_subscription = FakeSubscription(frames)

        _run_loop(runtime._wake_loop, self)

        self.assertEqual(detector.detections, 0)
        self.assertFalse(runtime._pending_wake_confirmed)

    def test_wake_loop_uses_higher_threshold_for_idle_wake_detection(self) -> None:
        config = _wake_config(rms_threshold=100, wake_rms_threshold=1000)
        controller = DummyController()
        runtime = _build_pipeline_runtime(controller, config)
        detector = StubWakeDetector()
        detector.should_detect = True
        runtime._wake_detector = detector
        frames = [
            AudioFrame(data=_frame_at_rms(config.frame_samples, 600), sequence=offset + 1)
            for offset in range(20)
        ]
        for frame in frames:
            runtime._ring_buffer.append(frame)
        runtime._wake_subscription = FakeSubscription(frames)

        _run_loop(runtime._wake_loop, self)

        self.assertEqual(detector.polls, 0)
        self.assertFalse(runtime._pending_wake_confirmed)

    def test_wake_loop_does_not_poll_during_active_conversation(self) -> None:
        """Follow-up speech is handled by the capture path, not wake detection."""
        config = _wake_config()
        controller = DummyController()
        controller.status.active_conversation = True
        controller.status.phase = AssistantPhase.AWAITING_USER_RESPONSE
        runtime = _build_pipeline_runtime(controller, config)
        detector = StubWakeDetector()
        detector.should_detect = True
        runtime._wake_detector = detector
        frames = _speech_run(config, count=8)
        for frame in frames:
            runtime._ring_buffer.append(frame)
        runtime._wake_subscription = FakeSubscription(frames)

        _run_loop(runtime._wake_loop, self)

        self.assertEqual(detector.polls, 0)
        self.assertEqual(detector.detections, 0)
        self.assertFalse(runtime._pending_wake_confirmed)

    def test_wake_loop_ignores_noise_that_never_fills_the_window(self) -> None:
        """Speech has to be dense inside the window, not merely spread across it.

        Counting speech frames cumulatively lets noise that crosses the threshold
        once per hold period climb past the minimum and stay there, pinning the
        detector at its full poll rate on audio that never contained a word.
        """
        config = _wake_config(
            wake_window_ms=1200,
            wake_hold_ms=450,
            wake_min_speech_ms=150,
            wake_poll_ms=300,
        )

        def polls_for(is_loud: Callable[[int], bool], count: int = 400) -> int:
            controller = DummyController()
            runtime = _build_pipeline_runtime(controller, config)
            detector = StubWakeDetector()
            runtime._wake_detector = detector
            frames = [
                AudioFrame(
                    data=(
                        _loud_frame(config.frame_samples)
                        if is_loud(offset)
                        else _silent_frame(config.frame_samples)
                    ),
                    sequence=offset + 1,
                )
                for offset in range(count)
            ]
            for frame in frames:
                runtime._ring_buffer.append(frame)
            runtime._wake_subscription = FakeSubscription(frames)
            _run_loop(runtime._wake_loop, self)
            return detector.polls

        # A loud frame every 14 frames stays inside the 15 frame hold, so the hold
        # never expires, yet a 40 frame window holds at most 3 of the 5 required.
        self.assertEqual(polls_for(lambda offset: offset % 14 == 0), 0)
        # Continuous speech still has to reach the detector.
        self.assertGreater(polls_for(lambda _offset: True), 0)

    def test_wake_loop_skips_inference_for_short_window(self) -> None:
        config = _wake_config(wake_min_window_ms=300)
        controller = DummyController()
        runtime = _build_pipeline_runtime(controller, config)
        detector = StubWakeDetector()
        detector.should_detect = True
        runtime._wake_detector = detector
        runtime._ring_buffer.append(AudioFrame(data=_loud_frame(config.frame_samples), sequence=1))

        runtime._wake_subscription = FakeSubscription(_speech_run(config, count=6))

        _run_loop(runtime._wake_loop, self)

        self.assertEqual(detector.detections, 0)
        self.assertFalse(runtime._pending_wake_confirmed)

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
                wake_confirmed: bool = False,
            ) -> str | None:
                result = super().process_transcript(
                    transcript,
                    utterance_id=utterance_id,
                    source_path=source_path,
                    wake_confirmed=wake_confirmed,
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

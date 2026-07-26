from __future__ import annotations

import sys
import subprocess
import threading
import time
import types
import unittest
from unittest import mock
from pathlib import Path

# Imported before any patch.dict(sys.modules) so teardown cannot evict it. numpy's C
# extension refuses to load twice in one process, which surfaces as an unrelated
# ImportError in whichever test happens to import it next.
import numpy  # noqa: F401

import agents.amy.modalities.audio.models as models_module
from agents.amy.modalities.audio import (
    AudioConfig,
    AudioCuePlayer,
    AudioFrame,
    AudioRingBuffer,
    AudioStream,
    LocalSpeaker,
    MicrophoneSource,
    MlxWhisperTranscriber,
    CommandRecorder,
    EnergyVad,
    RingSnapshot,
    SpeechTracker,
    VadState,
    WakeDetection,
    WhisperWakeDetector,
    WakeSession,
)


def _silent_frame(samples: int) -> bytes:
    return (b"\x00\x00") * samples


def _loud_frame(samples: int) -> bytes:
    return (b"\xff\x7f") * samples


def _matched(detection: WakeDetection) -> bool:
    return detection.matched


class EnergyVadTests(unittest.TestCase):
    def test_observes_speech_and_silence(self) -> None:
        config = AudioConfig(frame_ms=30, silence_ms=60, rms_threshold=100)
        vad = EnergyVad(config)

        state = vad.observe(AudioFrame(data=_silent_frame(config.frame_samples)))
        self.assertFalse(state.is_speech)
        self.assertEqual(state.silence_frames, 1)

        state = vad.observe(AudioFrame(data=_loud_frame(config.frame_samples)))
        self.assertTrue(state.is_speech)
        self.assertEqual(state.speech_frames, 1)

        state = vad.observe(AudioFrame(data=_silent_frame(config.frame_samples)))
        self.assertFalse(state.is_speech)
        self.assertEqual(state.silence_frames, 1)


    def test_rms_is_computed_once_and_shared_between_detectors(self) -> None:
        config = AudioConfig(frame_ms=30, rms_threshold=100)
        frame = AudioFrame(data=_loud_frame(config.frame_samples))
        capture_vad = EnergyVad(config)
        wake_vad = EnergyVad(config)

        self.assertTrue(capture_vad.observe(frame).is_speech)
        self.assertTrue(wake_vad.observe(frame).is_speech)

        # Both detectors read the same cached value rather than recomputing per frame.
        with mock.patch.object(models_module, "_compute_rms", side_effect=AssertionError) as recompute:
            self.assertEqual(frame.rms, 32767)
            recompute.assert_not_called()

    def test_rms_of_silence_is_zero(self) -> None:
        config = AudioConfig(frame_ms=30)

        self.assertEqual(AudioFrame(data=_silent_frame(config.frame_samples)).rms, 0)
        self.assertEqual(AudioFrame(data=b"").rms, 0)


class SpeechTrackerWindowTests(unittest.TestCase):
    def test_tracks_rolling_history(self) -> None:
        config = AudioConfig(frame_ms=30, wake_window_ms=90)
        tracker = SpeechTracker(hold_frames=1, window_frames=config.wake_window_frames)

        tracker.observe(sequence=1, is_speech=False)
        tracker.observe(sequence=2, is_speech=True)
        tracker.observe(sequence=3, is_speech=True)

        self.assertEqual(tracker.frame_count, 3)
        self.assertEqual(tracker.window_speech_frames, 2)
        self.assertTrue(tracker.window_full)
        self.assertAlmostEqual(tracker.speech_density, 2 / 3)

    def test_evicts_speech_from_the_front(self) -> None:
        config = AudioConfig(frame_ms=30, wake_window_ms=60)
        tracker = SpeechTracker(hold_frames=1, window_frames=config.wake_window_frames)

        tracker.observe(sequence=1, is_speech=True)
        tracker.observe(sequence=2, is_speech=False)
        tracker.observe(sequence=3, is_speech=False)

        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.window_speech_frames, 0)
        self.assertTrue(tracker.window_full)


class SpeechTrackerTests(unittest.TestCase):
    def test_reports_edges_and_hold_expiry(self) -> None:
        tracker = SpeechTracker(hold_frames=2)

        tracker.observe(is_speech=True)
        self.assertTrue(tracker.speech_started)
        self.assertTrue(tracker.speech_active)
        self.assertFalse(tracker.speech_finished)
        self.assertEqual(tracker.speech_frames, 1)
        self.assertEqual(tracker.consecutive_speech_frames, 1)

        tracker.observe(is_speech=True)
        self.assertFalse(tracker.speech_started)
        self.assertTrue(tracker.speech_active)
        self.assertFalse(tracker.speech_finished)
        self.assertEqual(tracker.speech_frames, 2)
        self.assertEqual(tracker.consecutive_speech_frames, 2)

        tracker.observe(is_speech=False)
        self.assertFalse(tracker.speech_started)
        self.assertTrue(tracker.speech_active)
        self.assertFalse(tracker.speech_finished)
        self.assertEqual(tracker.consecutive_speech_frames, 0)

        tracker.observe(is_speech=False)
        self.assertFalse(tracker.speech_finished)
        self.assertTrue(tracker.speech_active)

        tracker.observe(is_speech=False)
        self.assertTrue(tracker.speech_finished)
        self.assertFalse(tracker.speech_active)
        self.assertEqual(tracker.speech_frames, 0)


class WakeSessionTests(unittest.TestCase):
    def test_attempts_after_consecutive_speech_frames(self) -> None:
        config = AudioConfig(frame_ms=30, wake_min_speech_ms=90)
        session = WakeSession(config)

        for sequence in range(1, 3):
            session.observe(sequence=sequence, is_speech=True)
            self.assertFalse(session.should_attempt())

        session.observe(sequence=3, is_speech=True)

        self.assertTrue(session.should_attempt())

    def test_sparse_speech_does_not_accumulate_into_attempt(self) -> None:
        config = AudioConfig(frame_ms=30, wake_min_speech_ms=90)
        session = WakeSession(config)

        for sequence, is_speech in enumerate((True, False, True, False, True), start=1):
            session.observe(sequence=sequence, is_speech=is_speech)

        self.assertFalse(session.should_attempt())

    def test_trailing_attempt_only_after_missed_early_attempt(self) -> None:
        config = AudioConfig(frame_ms=30, wake_hold_ms=60, wake_min_speech_ms=60)
        session = WakeSession(config)

        session.observe(sequence=1, is_speech=True)
        session.observe(sequence=2, is_speech=True)
        self.assertTrue(session.should_attempt())
        session.record_attempt(matched=False)

        for sequence in range(3, 6):
            session.observe(sequence=sequence, is_speech=False)

        self.assertTrue(session.should_attempt_trailing())

    def test_retries_active_speech_after_poll_hop(self) -> None:
        config = AudioConfig(frame_ms=30, wake_min_speech_ms=60, wake_poll_ms=90)
        session = WakeSession(config)

        session.observe(sequence=1, is_speech=True)
        session.observe(sequence=2, is_speech=True)
        self.assertTrue(session.should_attempt())
        session.record_attempt(matched=False)

        session.observe(sequence=3, is_speech=True)
        session.observe(sequence=4, is_speech=True)
        self.assertFalse(session.should_attempt())

        session.observe(sequence=5, is_speech=True)
        self.assertTrue(session.should_attempt())

    def test_reset_clears_attempt_state_and_trackers(self) -> None:
        config = AudioConfig(frame_ms=30, wake_min_speech_ms=60)
        session = WakeSession(config)
        session.observe(sequence=1, is_speech=True)
        session.observe(sequence=2, is_speech=True)
        session.record_attempt(matched=True)

        session.reset()

        self.assertFalse(session.attempted)
        self.assertFalse(session.matched)
        self.assertFalse(session.speech_active)
        self.assertEqual(session.tracker.frame_count, 0)


class AudioRingBufferTests(unittest.TestCase):
    def test_snapshot_returns_recent_frames(self) -> None:
        config = AudioConfig(frame_ms=30, ring_buffer_ms=120)
        ring = AudioRingBuffer(config)
        for index in range(1, 7):
            ring.append(AudioFrame(data=bytes([index]), sequence=index))

        snapshot = ring.snapshot(duration_ms=60)

        self.assertEqual([frame.sequence for frame in snapshot.frames], [5, 6])
        self.assertEqual(snapshot.last_sequence, 6)

    def test_ring_buffer_evicts_old_frames(self) -> None:
        config = AudioConfig(frame_ms=30, ring_buffer_ms=60)
        ring = AudioRingBuffer(config)
        for index in range(1, 6):
            ring.append(AudioFrame(data=bytes([index]), sequence=index))

        snapshot = ring.snapshot()
        self.assertEqual([frame.sequence for frame in snapshot.frames], [4, 5])


class CommandRecorderTests(unittest.TestCase):
    def test_emits_segment_after_voice_and_silence(self) -> None:
        config = AudioConfig(frame_ms=30, silence_ms=60, rms_threshold=100)
        recorder = CommandRecorder(config)
        recorder.begin(
            RingSnapshot(
                frames=(
                    AudioFrame(data=_silent_frame(config.frame_samples), sequence=1),
                    AudioFrame(data=_silent_frame(config.frame_samples), sequence=2),
                ),
                last_sequence=2,
            )
        )

        self.assertIsNone(recorder.feed(AudioFrame(data=_loud_frame(config.frame_samples), sequence=3), VadState(True, 0, 1)))
        self.assertIsNone(recorder.feed(AudioFrame(data=_loud_frame(config.frame_samples), sequence=4), VadState(True, 0, 2)))
        self.assertIsNone(recorder.feed(AudioFrame(data=_silent_frame(config.frame_samples), sequence=5), VadState(False, 1, 2)))
        segment = recorder.feed(AudioFrame(data=_silent_frame(config.frame_samples), sequence=6), VadState(False, 2, 2))

        self.assertIsNotNone(segment)
        assert segment is not None
        self.assertIsInstance(segment.pcm, bytes)
        self.assertGreater(len(segment.pcm), 0)
        self.assertGreater(segment.duration_seconds, 0)

    def test_drops_duplicate_sequences(self) -> None:
        config = AudioConfig(frame_ms=30, silence_ms=60, rms_threshold=100)
        recorder = CommandRecorder(config)
        recorder.begin(RingSnapshot(frames=(), last_sequence=4))

        self.assertIsNone(recorder.feed(AudioFrame(data=_loud_frame(config.frame_samples), sequence=4), VadState(True, 0, 1)))
        self.assertTrue(recorder.is_recording)


class AudioStreamTests(unittest.TestCase):
    def test_fan_out_drops_oldest_for_slow_subscriber(self) -> None:
        class FakeMicrophoneSource:
            def __init__(self, _config: AudioConfig) -> None:
                self._frames = [
                    AudioFrame(data=b"1", sequence=0),
                    AudioFrame(data=b"2", sequence=0),
                    AudioFrame(data=b"3", sequence=0),
                ]

            def __enter__(self) -> "FakeMicrophoneSource":
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

            def frames(self) -> list[AudioFrame]:
                return self._frames

        stream = AudioStream(AudioConfig(), microphone_factory=FakeMicrophoneSource)
        slow = stream.subscribe("slow", maxsize=1)
        fast = stream.subscribe("fast", maxsize=8)
        stream.start()
        assert stream._thread is not None
        stream._thread.join(timeout=1)
        stream.stop()

        self.assertGreaterEqual(slow.dropped_frames, 1)
        slow_frame = slow.receive(timeout=0.1)
        fast_frame = fast.receive(timeout=0.1)
        self.assertIsNotNone(slow_frame)
        self.assertIsNotNone(fast_frame)
        assert slow_frame is not None and fast_frame is not None
        self.assertEqual(slow_frame.data, b"3")
        self.assertEqual(fast_frame.data, b"1")

    def test_microphone_failure_is_reported_and_closes_subscribers(self) -> None:
        class ExplodingMicrophoneSource:
            def __init__(self, _config: AudioConfig) -> None:
                return None

            def __enter__(self) -> "ExplodingMicrophoneSource":
                raise OSError("device disconnected")

            def __exit__(self, *_exc: object) -> None:
                return None

        errors: list[Exception] = []
        stream = AudioStream(
            AudioConfig(),
            microphone_factory=ExplodingMicrophoneSource,
            on_error=errors.append,
        )
        subscription = stream.subscribe("pipeline", maxsize=4)
        stream.start()
        assert stream._thread is not None
        stream._thread.join(timeout=1)

        self.assertEqual(len(errors), 1)
        self.assertIn("device disconnected", str(errors[0]))
        self.assertTrue(subscription.is_closed)
        self.assertIsNone(subscription.receive(timeout=0.05))


class WhisperWakeDetectorTests(unittest.TestCase):
    def test_warmup_and_alias_matching(self) -> None:
        calls: list[tuple[object, dict[str, object]]] = []

        def fake_transcribe(audio: object, **kwargs: object) -> dict[str, str]:
            calls.append((audio, dict(kwargs)))
            return {"text": "Hey, Amy"}

        fake_module = types.SimpleNamespace(transcribe=fake_transcribe)
        with mock.patch.dict(sys.modules, {"mlx_whisper": fake_module}):
            detector = WhisperWakeDetector(AudioConfig())
            detector.warmup()
            self.assertEqual(len(calls), 1)
            self.assertTrue(_matched(detector.detect(_loud_frame(16_000), vad_state=VadState(True, 0, 1))))
            self.assertEqual(len(calls), 2)

    def test_matches_wake_word_followed_by_the_command(self) -> None:
        """The command usually arrives in the same breath as the wake word."""
        detector = WhisperWakeDetector(AudioConfig())

        for text in (
            "amy",
            "hey amy",
            "hi amy",
            "amy can you",
            "amy what is the weather",
            "hey amy can you tell me what the weather is",
            "emmy can you",
        ):
            self.assertTrue(detector._matches_wake_word(text), text)

    def test_rejects_wake_word_that_is_not_leading(self) -> None:
        detector = WhisperWakeDetector(AudioConfig())

        for text in (
            "",
            "tell me about amy",
            "amybody home",
            "can you tell me what the weather is",
            "in west palm beach today",
        ):
            self.assertFalse(detector._matches_wake_word(text), text)

    def test_ignores_speech_that_is_not_the_wake_word(self) -> None:
        fake_module = types.SimpleNamespace(
            transcribe=lambda audio, **kwargs: {"text": "what is the weather tomorrow"}
        )
        with mock.patch.dict(sys.modules, {"mlx_whisper": fake_module}):
            detector = WhisperWakeDetector(AudioConfig())

            self.assertFalse(_matched(detector.detect(_loud_frame(16_000), vad_state=VadState(True, 0, 1))))

    def test_cooldown_expires_on_wall_clock_not_call_count(self) -> None:
        fake_module = types.SimpleNamespace(transcribe=lambda audio, **kwargs: {"text": "amy"})
        with mock.patch.dict(sys.modules, {"mlx_whisper": fake_module}):
            detector = WhisperWakeDetector(AudioConfig(wake_cooldown_ms=40))
            speech = VadState(True, 0, 1)

            self.assertTrue(_matched(detector.detect(_loud_frame(16_000), vad_state=speech)))
            # Repeated polls inside the window stay suppressed regardless of how many run.
            for _ in range(50):
                self.assertFalse(_matched(detector.detect(_loud_frame(16_000), vad_state=speech)))

            time.sleep(0.05)
            self.assertTrue(_matched(detector.detect(_loud_frame(16_000), vad_state=speech)))

    def test_skips_inference_while_vad_reports_silence(self) -> None:
        calls: list[object] = []

        def fake_transcribe(audio: object, **kwargs: object) -> dict[str, str]:
            calls.append(audio)
            return {"text": "amy"}

        fake_module = types.SimpleNamespace(transcribe=fake_transcribe)
        with mock.patch.dict(sys.modules, {"mlx_whisper": fake_module}):
            detector = WhisperWakeDetector(AudioConfig())

            self.assertFalse(_matched(detector.detect(_silent_frame(16_000), vad_state=VadState(False, 5, 0))))
            self.assertEqual(calls, [])

    def test_detection_reports_whether_anything_followed_the_wake_word(self) -> None:
        """A spoken greeting is long enough to talk over a command, so bare matters."""
        for text, expect_bare, expect_trailing in (
            ("amy", True, ""),
            ("hey amy", True, ""),
            ("amy what is the weather", False, "what is the weather"),
            ("hey amy can you", False, "can you"),
        ):
            fake_module = types.SimpleNamespace(
                transcribe=lambda audio, _t=text, **kwargs: {"text": _t}
            )
            with mock.patch.dict(sys.modules, {"mlx_whisper": fake_module}):
                detector = WhisperWakeDetector(AudioConfig())
                detection = detector.detect(_loud_frame(16_000), vad_state=VadState(True, 0, 1))

            self.assertTrue(detection.matched, text)
            self.assertEqual(detection.trailing_text, expect_trailing, text)
            self.assertEqual(detection.is_bare, expect_bare, text)

    def test_supports_custom_wake_word(self) -> None:
        detector = WhisperWakeDetector(AudioConfig(), wake_word="jarvis")

        self.assertTrue(detector._matches_wake_word("jarvis what time is it"))
        self.assertFalse(detector._matches_wake_word("amy what time is it"))


class FakeOutputStream:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


class AudioCuePlayerTests(unittest.TestCase):
    _CUE_PCM = (b"\x01\x00") * 512

    def _player(self) -> tuple[AudioCuePlayer, list[FakeOutputStream]]:
        streams: list[FakeOutputStream] = []

        def factory(**kwargs: object) -> FakeOutputStream:
            stream = FakeOutputStream(**kwargs)
            streams.append(stream)
            return stream

        return AudioCuePlayer(AudioConfig(), stream_factory=factory, pcm=self._CUE_PCM), streams

    def _drain(self, player: AudioCuePlayer, frames: int = 256, rounds: int = 200) -> bytes:
        rendered = bytearray()
        for _ in range(rounds):
            block = bytearray(frames * 2)
            player._fill(block, frames, None, None)
            rendered.extend(block)
            if not player.is_playing.is_set():
                break
        return bytes(rendered)

    def test_player_uses_injected_pcm(self) -> None:
        player, streams = self._player()

        self.assertEqual(player.pcm, self._CUE_PCM)
        player.start()

        self.assertEqual(len(streams), 1)
        self.assertTrue(streams[0].started)
        self.assertEqual(streams[0].kwargs["samplerate"], 16_000)
        self.assertEqual(streams[0].kwargs["dtype"], "int16")

    def test_play_streams_the_cue_then_falls_silent(self) -> None:
        player, streams = self._player()
        player.start()

        self.assertEqual(len(streams), 1)
        self.assertTrue(streams[0].started)
        self.assertEqual(streams[0].kwargs["samplerate"], 16_000)
        self.assertEqual(streams[0].kwargs["dtype"], "int16")

        # Nothing plays until asked, even though the stream is already open.
        self.assertEqual(self._drain(player, rounds=1).strip(b"\x00"), b"")

        player.play()
        self.assertTrue(player.is_playing.is_set())
        rendered = self._drain(player)

        self.assertFalse(player.is_playing.is_set())
        self.assertTrue(rendered.startswith(player.pcm))

    def test_play_is_a_no_op_when_the_device_is_unavailable(self) -> None:
        def exploding_factory(**_kwargs: object) -> FakeOutputStream:
            raise OSError("no output device")

        player = AudioCuePlayer(AudioConfig(), stream_factory=exploding_factory)
        player.start()

        player.play()

        self.assertFalse(player.is_playing.is_set())
        player.close()

    def test_close_releases_the_stream(self) -> None:
        player, streams = self._player()
        player.start()

        player.close()

        self.assertTrue(streams[0].closed)
        self.assertFalse(streams[0].started)


class MlxWhisperTranscriberTests(unittest.TestCase):
    def test_warmup_loads_model_once(self) -> None:
        calls: list[tuple[object, dict[str, object]]] = []

        def fake_transcribe(audio: object, **kwargs: object) -> dict[str, str]:
            calls.append((audio, dict(kwargs)))
            return {"text": " hello "}

        fake_module = types.SimpleNamespace(transcribe=fake_transcribe)
        with mock.patch.dict(sys.modules, {"mlx_whisper": fake_module}):
            transcriber = MlxWhisperTranscriber(
                model_repo="mlx-community/whisper-large-v3-turbo",
                language="en",
            )
            transcriber.warmup()
            transcriber.warmup()

            self.assertEqual(len(calls), 1)
            self.assertEqual(
                calls[0][1],
                {
                    "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
                    "language": "en",
                    "verbose": None,
                    "temperature": 0.0,
                    "condition_on_previous_text": False,
                    "word_timestamps": False,
                },
            )
            self.assertEqual(transcriber.transcribe(Path("/tmp/example.wav")), "hello")
            self.assertEqual(len(calls), 2)
            self.assertEqual(
                calls[1][1],
                {
                    "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
                    "language": "en",
                    "verbose": None,
                    "temperature": 0.0,
                    "condition_on_previous_text": False,
                    "word_timestamps": False,
                },
            )


class LocalSpeakerTests(unittest.TestCase):
    def test_stop_handles_hung_process_timeout(self) -> None:
        speaker = LocalSpeaker()
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired(cmd="say", timeout=1), None]
        speaker._process = process  # type: ignore[attr-defined]
        speaker.is_speaking.set()

        speaker.stop()

        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertFalse(speaker.is_speaking.is_set())

    def test_overlapping_speaks_keep_latest_flag_set(self) -> None:
        speaker = LocalSpeaker()
        first_started = threading.Event()
        second_started = threading.Event()
        first_released = threading.Event()
        second_released = threading.Event()

        class FakeProcess:
            def __init__(self, started: threading.Event, released: threading.Event) -> None:
                self._started = started
                self._released = released
                self.terminated = False
                self.killed = False

            def wait(self, timeout: float | None = None) -> None:
                self._started.set()
                self._released.wait(timeout=timeout)

            def poll(self) -> int | None:
                return None if not self._released.is_set() else 0

            def terminate(self) -> None:
                self.terminated = True
                self._released.set()

            def kill(self) -> None:
                self.killed = True
                self._released.set()

        processes: list[FakeProcess] = [
            FakeProcess(first_started, first_released),
            FakeProcess(second_started, second_released),
        ]

        def fake_popen(*_args: object, **_kwargs: object) -> FakeProcess:
            return processes.pop(0)

        with (
            mock.patch("agents.amy.modalities.audio.tts.platform.system", return_value="Darwin"),
            mock.patch("agents.amy.modalities.audio.tts.subprocess.Popen", side_effect=fake_popen),
        ):
            first = threading.Thread(target=speaker.speak, args=("first",))
            first.start()
            self.assertTrue(first_started.wait(timeout=1))

            second = threading.Thread(target=speaker.speak, args=("second",))
            second.start()
            self.assertTrue(second_started.wait(timeout=1))

            first.join(timeout=1)
            self.assertFalse(first.is_alive())
            self.assertTrue(speaker.is_speaking.is_set())

            second_released.set()
            second.join(timeout=1)
            self.assertFalse(second.is_alive())
            self.assertFalse(speaker.is_speaking.is_set())


class MicrophoneSourceTests(unittest.TestCase):
    def test_microphone_source_passes_selected_device_to_sounddevice(self) -> None:
        captured: dict[str, object] = {}

        class FakeStream:
            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def close(self) -> None:
                return None

            def read(self, frames: int) -> tuple[bytes, bool]:
                return (b"\x00\x00" * frames, True)

        def fake_raw_input_stream(**kwargs: object) -> FakeStream:
            captured.update(kwargs)
            return FakeStream()

        fake_sounddevice = types.SimpleNamespace(RawInputStream=fake_raw_input_stream)
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sounddevice}):
            source = MicrophoneSource(AudioConfig(input_device="Audient iD24"))
            with source:
                frame = next(source.frames())

        self.assertEqual(captured["device"], "Audient iD24")
        self.assertEqual(captured["channels"], 1)
        self.assertEqual(captured["dtype"], "int16")
        self.assertIsInstance(frame, AudioFrame)
        expected_frame = b"\x00\x00" * AudioConfig(input_device="Audient iD24").frame_samples
        self.assertEqual(frame.data, expected_frame)
        self.assertTrue(frame.overflow)

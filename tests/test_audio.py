from __future__ import annotations

import sys
import subprocess
import types
import unittest
from unittest import mock
from pathlib import Path

from agents.amy.modalities.audio import (
    AudioConfig,
    AudioFrame,
    LocalSpeaker,
    MicrophoneSource,
    MlxWhisperTranscriber,
    SpeechSegmenter,
)


def _silent_frame(samples: int) -> bytes:
    return (b"\x00\x00") * samples


def _loud_frame(samples: int) -> bytes:
    return (b"\xff\x7f") * samples


class SpeechSegmenterTests(unittest.TestCase):
    def test_emits_segment_after_voice_and_silence(self) -> None:
        config = AudioConfig(frame_ms=30, silence_ms=60, rms_threshold=100)
        segmenter = SpeechSegmenter(config)

        for _ in range(config.pre_roll_frames):
            self.assertIsNone(segmenter.feed(_silent_frame(config.frame_samples)))

        self.assertIsNone(segmenter.feed(_loud_frame(config.frame_samples)))
        self.assertIsNone(segmenter.feed(_loud_frame(config.frame_samples)))
        self.assertIsNone(segmenter.feed(_silent_frame(config.frame_samples)))
        segment = segmenter.feed(_silent_frame(config.frame_samples))

        self.assertIsNotNone(segment)
        assert segment is not None
        self.assertIsInstance(segment.pcm, bytes)
        self.assertGreater(len(segment.pcm), 0)
        self.assertGreater(segment.duration_seconds, 0)

    def test_emits_segment_after_max_utterance_duration(self) -> None:
        config = AudioConfig(
            frame_ms=30,
            pre_roll_ms=30,
            silence_ms=9_999,
            max_utterance_ms=60,
            rms_threshold=100,
        )
        segmenter = SpeechSegmenter(config)

        self.assertIsNone(segmenter.feed(_silent_frame(config.frame_samples)))
        self.assertIsNone(segmenter.feed(_loud_frame(config.frame_samples)))
        segment = segmenter.feed(_loud_frame(config.frame_samples))

        self.assertIsNotNone(segment)
        assert segment is not None
        self.assertGreater(len(segment.pcm), 0)
        self.assertGreater(segment.duration_seconds, 0)


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
                    "verbose": False,
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
                    "verbose": False,
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

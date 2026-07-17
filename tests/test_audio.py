from __future__ import annotations

import sys
import types
import unittest
from unittest import mock
from pathlib import Path

from agents.amy.modalities.audio import AudioConfig, FasterWhisperTranscriber, SpeechSegmenter


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
        self.assertEqual(segment.path.suffix, ".wav")
        self.assertGreater(segment.duration_seconds, 0)
        self.assertTrue(segment.path.exists())


class FasterWhisperTranscriberTests(unittest.TestCase):
    def test_warmup_loads_model_once(self) -> None:
        calls: list[tuple[object, ...]] = []

        class FakeModel:
            def __init__(self, model_name: str, device: str, compute_type: str) -> None:
                calls.append((model_name, device, compute_type))

            def transcribe(
                self,
                audio: str,
                *,
                language: str | None,
                beam_size: int,
                vad_filter: bool,
            ) -> tuple[list[object], object]:
                return ([], object())

        fake_module = types.SimpleNamespace(WhisperModel=FakeModel)
        with mock.patch.dict(sys.modules, {"faster_whisper": fake_module}):
            transcriber = FasterWhisperTranscriber(model_name="tiny", language="en")
            transcriber.warmup()
            transcriber.warmup()

            self.assertEqual(calls, [("tiny", "cpu", "int8")])
            self.assertEqual(transcriber.transcribe(Path("/tmp/example.wav")), "")
            self.assertEqual(calls, [("tiny", "cpu", "int8")])

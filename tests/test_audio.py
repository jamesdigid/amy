from __future__ import annotations

import sys
import types
import unittest
from unittest import mock
from pathlib import Path

from agents.amy.modalities.audio import AudioConfig, MlxWhisperTranscriber, SpeechSegmenter


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

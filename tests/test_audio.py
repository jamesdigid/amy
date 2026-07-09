from __future__ import annotations

import unittest

from amy.audio import AudioConfig, SpeechSegmenter


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

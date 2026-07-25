from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging
import os
import queue
import subprocess
import threading
import time
from enum import Enum
from typing import Callable, TypeVar

from ..models import AssistantPhase
from ..modalities.audio import AudioConfig, MicrophoneSource, SpeechSegmenter, Transcriber
from ..controller import AssistantController
from .status import AmyStatusReporter


StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class UtteranceJob:
    utterance_id: str
    epoch: int
    captured_at: float
    pcm: bytes
    duration_seconds: float


@dataclass(frozen=True)
class TranscriptJob:
    utterance_id: str
    epoch: int
    captured_at: float
    transcript: str


class RuntimeState(str, Enum):
    NEW = "new"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


logger = logging.getLogger(__name__)
_QueueItem = TypeVar("_QueueItem")


@dataclass
class AssistantRuntime:
    controller: AssistantController
    transcriber: Transcriber
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    microphone_factory: Callable[[AudioConfig], MicrophoneSource] = field(default_factory=lambda: MicrophoneSource)
    segmenter_factory: Callable[[AudioConfig], SpeechSegmenter] = field(default_factory=lambda: SpeechSegmenter)
    log_transcripts: bool = False
    status_reporter: AmyStatusReporter | None = None
    on_status: StatusCallback = field(default=lambda _message: None)
    _utterance_queue: queue.Queue[UtteranceJob] = field(
        default_factory=lambda: queue.Queue[UtteranceJob](maxsize=4), init=False
    )
    _transcript_queue: queue.Queue[TranscriptJob] = field(
        default_factory=lambda: queue.Queue[TranscriptJob](maxsize=4), init=False
    )
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _capture_enabled: threading.Event = field(default_factory=threading.Event, init=False)
    _capture_thread: threading.Thread | None = field(default=None, init=False)
    _stt_thread: threading.Thread | None = field(default=None, init=False)
    _worker_thread: threading.Thread | None = field(default=None, init=False)
    _acknowledgement_thread: threading.Thread | None = field(default=None, init=False)
    _acknowledgement_stop: threading.Event = field(default_factory=threading.Event, init=False)
    _acknowledgement_sound_path: str = field(init=False)
    _acknowledgement_process: subprocess.Popen[bytes] | None = field(default=None, init=False)
    _runtime_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _runtime_state: RuntimeState = field(default=RuntimeState.NEW, init=False)
    _capture_epoch: int = field(default=0, init=False)
    _utterance_counter: int = field(default=0, init=False)
    _utterance_dropped_count: int = field(default=0, init=False)
    _transcript_dropped_count: int = field(default=0, init=False)
    _capture_overflow_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._acknowledgement_sound_path = str(
            Path(__file__).resolve().parent / ".." / "assets" / "Glass.aiff"
        )

    def start(self) -> None:
        with self._runtime_lock:
            if self._runtime_state is RuntimeState.STOPPED:
                raise RuntimeError("runtime cannot be restarted after stop")
            if self._runtime_state is RuntimeState.RUNNING:
                return
            self._stop_event.clear()
            self._acknowledgement_stop.clear()
            self._capture_enabled.set()
            self._runtime_state = RuntimeState.RUNNING
            self._capture_epoch += 1

        warmup = getattr(self.transcriber, "warmup", None)
        if callable(warmup):
            logger.debug("warming transcription model")
            warmup()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._stt_thread = threading.Thread(target=self._stt_worker_loop, daemon=True)
        self._stt_thread.start()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self.on_status("runtime started")

    def pause_capture(self) -> None:
        logger.debug("pause_capture requested")
        self.controller.pause()
        self.on_status("speech interrupted")

    def resume_capture(self) -> None:
        if self._stop_event.is_set():
            return
        logger.debug("resume_capture requested")
        self.controller.resume()
        if not self._capture_enabled.is_set():
            self._capture_enabled.set()
            if self._capture_thread is None or not self._capture_thread.is_alive():
                self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self._capture_thread.start()
        self.on_status("capture resumed")

    def stop(self) -> None:
        with self._runtime_lock:
            if self._runtime_state is RuntimeState.STOPPED:
                return
            self._runtime_state = RuntimeState.STOPPING
            self._stop_event.set()
            self._acknowledgement_stop.set()
            self._capture_enabled.clear()

        self._stop_acknowledgement_process()
        self.controller.stop()
        self.on_status("runtime stopping")
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2)
        if self._stt_thread is not None and self._stt_thread.is_alive():
            self._stt_thread.join(timeout=2)
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2)
        if self._acknowledgement_thread is not None and self._acknowledgement_thread.is_alive():
            self._acknowledgement_thread.join(timeout=2)
        with self._runtime_lock:
            self._runtime_state = RuntimeState.STOPPED

    def status_text(self) -> str:
        status = self.controller.get_status()
        if self.status_reporter is not None:
            return self.status_reporter.build_report(status)
        return (
            f"phase={status.phase.value} "
            f"active={status.active_conversation} "
            f"paused={status.paused} "
            f"last_user={status.last_user_text!r} "
            f"last_assistant={status.last_assistant_text!r}"
        )

    def _capture_loop(self) -> None:
        speech_segmenter = self.segmenter_factory(self.audio_config)
        try:
            with self.microphone_factory(self.audio_config) as microphone:
                for frame in microphone.frames():
                    if self._stop_event.is_set() or not self._capture_enabled.is_set():
                        break
                    if frame.overflow:
                        self._capture_overflow_count += 1
                        logger.warning("audio overflow detected (%d)", self._capture_overflow_count)
                        speech_segmenter.reset()
                        continue
                    if self.controller.should_drop_main_transcript():
                        speech_segmenter.reset()
                        continue

                    segment = speech_segmenter.feed(frame.data)
                    if segment is None:
                        continue
                    utterance_id = self._next_utterance_id()
                    utterance = UtteranceJob(
                        utterance_id=utterance_id,
                        epoch=self.controller.epoch,
                        captured_at=time.monotonic(),
                        pcm=segment.pcm,
                        duration_seconds=segment.duration_seconds,
                    )
                    self._enqueue_utterance(utterance)
        except Exception as exc:  # pragma: no cover - runtime path
            self.controller.status.error_message = str(exc)
            self.on_status(f"capture error: {exc}")
        finally:
            self.on_status("capture loop exited")

    def _stt_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._utterance_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if self._is_job_stale(job.epoch, job.captured_at):
                continue

            wav_path = self._write_temp_wav(job.pcm)
            try:
                transcript_started = time.perf_counter()
                try:
                    text = self.transcriber.transcribe(wav_path)
                except Exception as exc:  # pragma: no cover - runtime path
                    self.controller.status.error_message = str(exc)
                    self.on_status(f"stt worker error: {exc}")
                    continue
                transcript_elapsed = time.perf_counter() - transcript_started
                logger.debug(
                    "transcript ready in %.3fs for %.2fs audio from %s",
                    transcript_elapsed,
                    job.duration_seconds,
                    wav_path,
                )
            finally:
                self._cleanup_audio_segment(wav_path)

            if self._is_job_stale(job.epoch, job.captured_at):
                continue

            transcript_job = TranscriptJob(
                utterance_id=job.utterance_id,
                epoch=job.epoch,
                captured_at=job.captured_at,
                transcript=text,
            )
            self._enqueue_transcript(transcript_job)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._transcript_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not job.transcript.strip():
                continue
            if self._is_job_stale(job.epoch, job.captured_at):
                continue
            try:
                worker_started = time.perf_counter()
                self.controller.process_transcript(job.transcript, utterance_id=job.utterance_id)
                worker_elapsed = time.perf_counter() - worker_started
                logger.debug("controller processed transcript in %.3fs", worker_elapsed)
            except Exception as exc:  # pragma: no cover - runtime path
                self.controller.status.error_message = str(exc)
                self.on_status(f"worker error: {exc}")
            finally:
                time.sleep(0.01)

    def _enqueue_utterance(self, job: UtteranceJob) -> None:
        self._enqueue_with_drop_oldest(self._utterance_queue, job, "utterance")

    def _enqueue_transcript(self, job: TranscriptJob) -> None:
        self._enqueue_with_drop_oldest(self._transcript_queue, job, "transcript")

    def _enqueue_with_drop_oldest(self, queue_obj: queue.Queue[_QueueItem], item: _QueueItem, label: str) -> None:
        try:
            queue_obj.put_nowait(item)
        except queue.Full:
            try:
                queue_obj.get_nowait()
            except queue.Empty:
                pass
            queue_obj.put_nowait(item)
            if label == "utterance":
                self._utterance_dropped_count += 1
                logger.warning("dropped oldest utterance (%d)", self._utterance_dropped_count)
            else:
                self._transcript_dropped_count += 1
                logger.warning("dropped oldest transcript (%d)", self._transcript_dropped_count)

    def _should_queue_main_transcript(self) -> bool:
        speaker_state = getattr(self.controller.speaker, "is_speaking", None)
        is_speaking = bool(speaker_state.is_set()) if speaker_state is not None else False
        if is_speaking:
            return False

        if self.controller.should_drop_main_transcript():
            return False

        status = self.controller.get_status()
        return status.phase not in {AssistantPhase.PAUSED}

    def _next_utterance_id(self) -> str:
        self._utterance_counter += 1
        return f"utt-{self._utterance_counter}"

    def _is_job_stale(self, epoch: int, captured_at: float) -> bool:
        if epoch != self.controller.epoch:
            return True
        return (time.monotonic() - captured_at) > 8.0

    def _write_temp_wav(self, audio: bytes) -> Path:
        import tempfile
        import wave

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.audio_config.sample_rate)
            wav_file.writeframes(audio)
        return temp_path

    def _log_transcript(
        self,
        _source: str,
        transcript: str,
        source_path: Path,
    ) -> None:
        if not self.log_transcripts:
            return
        logger.info("%s from %s", transcript, source_path)

    def play_acknowledgement_loop(self) -> None:
        if self._acknowledgement_thread is not None and self._acknowledgement_thread.is_alive():
            return

        self._acknowledgement_stop.clear()
        self._acknowledgement_thread = threading.Thread(target=self._acknowledgement_loop, daemon=True)
        self._acknowledgement_thread.start()

    def stop_acknowledgement_loop(self) -> None:
        self._acknowledgement_stop.set()
        if self._acknowledgement_thread is not None and self._acknowledgement_thread.is_alive():
            self._acknowledgement_thread.join(timeout=2)

    def _acknowledgement_loop(self) -> None:
        self.on_status("Amy is looking")
        self._play_acknowledgement_sound()
        if self._acknowledgement_stop.wait(timeout=0.7):
            return
        while not self._acknowledgement_stop.wait(timeout=1.2):
            self._play_acknowledgement_sound()
            self.on_status("Amy is looking")

    def _play_acknowledgement_sound(self) -> None:
        if not os.path.exists(self._acknowledgement_sound_path):
            return

        try:
            process = subprocess.Popen(
                ["afplay", self._acknowledgement_sound_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._acknowledgement_process = process
            process.wait()
        except Exception:
            self.on_status("Amy is looking")
        finally:
            self._acknowledgement_process = None

    def _stop_acknowledgement_process(self) -> None:
        process = self._acknowledgement_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except Exception:
                pass
        self._acknowledgement_process = None

    def _cleanup_audio_segment(self, audio_path: Path) -> None:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("failed to clean up audio segment: %s", audio_path, exc_info=True)

__all__ = ["AssistantRuntime"]

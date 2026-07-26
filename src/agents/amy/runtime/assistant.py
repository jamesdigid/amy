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

from ..modalities.audio import (
    AudioConfig,
    AudioCuePlayer,
    AudioRingBuffer,
    AudioStream,
    CommandRecorder,
    EnergyVad,
    MicrophoneSource,
    FrameSubscription,
    RingSnapshot,
    Transcriber,
    WakeDetector,
    WakeSession,
    StubWakeDetector,
)
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
    wake_confirmed: bool = False


@dataclass(frozen=True)
class TranscriptJob:
    utterance_id: str
    epoch: int
    captured_at: float
    transcript: str
    wake_confirmed: bool = False


class RuntimeState(str, Enum):
    NEW = "new"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


logger = logging.getLogger(__name__)
_QueueItem = TypeVar("_QueueItem")

_FRAME_QUEUE_MAXSIZE = 32


@dataclass
class AssistantRuntime:
    controller: AssistantController
    transcriber: Transcriber
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    microphone_factory: Callable[[AudioConfig], MicrophoneSource] = field(default_factory=lambda: MicrophoneSource)
    audio_stream_factory: Callable[[AudioConfig], AudioStream] | None = None
    ring_buffer_factory: Callable[[AudioConfig], AudioRingBuffer] = field(
        default_factory=lambda: AudioRingBuffer
    )
    vad_factory: Callable[[AudioConfig], EnergyVad] = field(default_factory=lambda: EnergyVad)
    wake_detector_factory: Callable[[AudioConfig], WakeDetector] = field(
        default_factory=lambda: lambda config: StubWakeDetector()
    )
    recorder_factory: Callable[[AudioConfig], CommandRecorder] = field(
        default_factory=lambda: CommandRecorder
    )
    cue_player_factory: Callable[[AudioConfig], AudioCuePlayer] | None = field(
        default_factory=lambda: AudioCuePlayer
    )
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
    _stt_thread: threading.Thread | None = field(default=None, init=False)
    _worker_thread: threading.Thread | None = field(default=None, init=False)
    _acknowledgement_thread: threading.Thread | None = field(default=None, init=False)
    _acknowledgement_stop: threading.Event = field(default_factory=threading.Event, init=False)
    _acknowledgement_sound_path: str = field(init=False)
    _acknowledgement_process: subprocess.Popen[bytes] | None = field(default=None, init=False)
    _runtime_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _runtime_state: RuntimeState = field(default=RuntimeState.NEW, init=False)
    _utterance_counter: int = field(default=0, init=False)
    _utterance_dropped_count: int = field(default=0, init=False)
    _transcript_dropped_count: int = field(default=0, init=False)
    _capture_overflow_count: int = field(default=0, init=False)
    _capture_gap_count: int = field(default=0, init=False)
    _wake_gap_count: int = field(default=0, init=False)
    _audio_stream: AudioStream | None = field(default=None, init=False)
    _pipeline_subscription: FrameSubscription | None = field(default=None, init=False)
    _wake_subscription: FrameSubscription | None = field(default=None, init=False)
    _pipeline_thread: threading.Thread | None = field(default=None, init=False)
    _wake_thread: threading.Thread | None = field(default=None, init=False)
    _ring_buffer: AudioRingBuffer | None = field(default=None, init=False)
    _vad: EnergyVad | None = field(default=None, init=False)
    _wake_vad: EnergyVad | None = field(default=None, init=False)
    _recorder: CommandRecorder | None = field(default=None, init=False)
    _wake_detector: WakeDetector | None = field(default=None, init=False)
    _cue_player: AudioCuePlayer | None = field(default=None, init=False)
    _pending_wake_snapshot: RingSnapshot | None = field(default=None, init=False)
    _pending_wake_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _pending_wake_confirmed: bool = field(default=False, init=False)
    _recording_wake_confirmed: bool = field(default=False, init=False)
    _capture_suspended: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._acknowledgement_sound_path = str(
            Path(__file__).resolve().parent / ".." / "assets" / "Glass.aiff"
        )

    def _build_audio_stream(self) -> AudioStream:
        if self.audio_stream_factory is not None:
            stream = self.audio_stream_factory(self.audio_config)
        else:
            stream = AudioStream(self.audio_config, microphone_factory=self.microphone_factory)
        stream.on_error = self._handle_audio_stream_error
        return stream

    def _frame_queue_maxsize(self) -> int:
        """Bound fan-out depth by the ring buffer so a lagging consumer cannot splice a gap.

        A wake handoff replays ring history for frames the pipeline has not reached
        yet and leans on the recorder's sequence guard to skip the overlap. That only
        holds while a late consumer stays inside the ring's window, so a queue deeper
        than the ring would stitch non-adjacent audio into one utterance.
        """
        return max(1, min(_FRAME_QUEUE_MAXSIZE, self.audio_config.ring_buffer_frames))

    def _handle_audio_stream_error(self, exc: Exception) -> None:
        self.controller.status.error_message = str(exc)
        self.on_status(f"audio stream error: {exc}")

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
            audio_stream = self._build_audio_stream()
            self._audio_stream = audio_stream
            self._ring_buffer = self.ring_buffer_factory(self.audio_config)
            self._vad = self.vad_factory(self.audio_config)
            self._wake_vad = self.vad_factory(self.audio_config)
            self._recorder = self.recorder_factory(self.audio_config)
            self._wake_detector = self.wake_detector_factory(self.audio_config)
            self._cue_player = (
                self.cue_player_factory(self.audio_config) if self.cue_player_factory is not None else None
            )
            self._pending_wake_snapshot = None
            self._pending_wake_confirmed = False
            self._recording_wake_confirmed = False
            self._capture_suspended = False

            frame_queue_maxsize = self._frame_queue_maxsize()
            audio_stream.add_observer(self._ring_buffer.append)
            self._pipeline_subscription = audio_stream.subscribe("pipeline", maxsize=frame_queue_maxsize)
            self._wake_subscription = audio_stream.subscribe("wake", maxsize=frame_queue_maxsize)

        warmup = getattr(self.transcriber, "warmup", None)
        if callable(warmup):
            logger.debug("warming transcription model")
            warmup()
        wake_warmup = getattr(self._wake_detector, "warmup", None)
        if callable(wake_warmup):
            logger.debug("warming wake detector")
            wake_warmup()
        if self._stop_event.is_set():
            audio_stream.stop()
            return
        audio_stream.start()
        if self._stop_event.is_set():
            audio_stream.stop()
            return
        if self._cue_player is not None:
            # Opened up front so the acknowledgement never pays device setup cost.
            self._cue_player.start()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._stt_thread = threading.Thread(target=self._stt_worker_loop, daemon=True)
        self._stt_thread.start()
        self._pipeline_thread = threading.Thread(target=self._pipeline_loop, daemon=True)
        self._pipeline_thread.start()
        self._wake_thread = threading.Thread(target=self._wake_loop, daemon=True)
        self._wake_thread.start()
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
        self.on_status("capture resumed")

    def stop(self) -> None:
        with self._runtime_lock:
            if self._runtime_state is RuntimeState.STOPPED:
                return
            self._runtime_state = RuntimeState.STOPPING
            self._stop_event.set()
            self._acknowledgement_stop.set()
            self._capture_enabled.clear()

        if self._audio_stream is not None:
            self._audio_stream.stop()
        if self._cue_player is not None:
            self._cue_player.close()
        self._stop_acknowledgement_process()
        self.controller.stop()
        self.on_status("runtime stopping")
        if self._pipeline_thread is not None and self._pipeline_thread.is_alive():
            self._pipeline_thread.join(timeout=2)
        if self._wake_thread is not None and self._wake_thread.is_alive():
            self._wake_thread.join(timeout=2)
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

    def _pipeline_loop(self) -> None:
        if self._pipeline_subscription is None:
            return
        assert self._ring_buffer is not None
        assert self._vad is not None
        assert self._recorder is not None
        pipeline_subscription = self._pipeline_subscription
        last_sequence = 0

        try:
            while not self._stop_event.is_set():
                frame = pipeline_subscription.receive(timeout=0.2)
                if frame is None:
                    if self._stop_event.is_set() or pipeline_subscription.is_closed:
                        break
                    continue
                if last_sequence and frame.sequence > last_sequence + 1:
                    self._capture_gap_count += 1
                    logger.warning(
                        "pipeline lost %d frame(s) before sequence %d (%d total)",
                        frame.sequence - last_sequence - 1,
                        frame.sequence,
                        self._capture_gap_count,
                    )
                    self._reset_capture_state()
                last_sequence = frame.sequence
                if frame.overflow:
                    self._capture_overflow_count += 1
                    logger.warning("audio overflow detected (%d)", self._capture_overflow_count)
                    self._reset_audio_state()
                    continue
                cue_is_playing = self._cue_player is not None and self._cue_player.is_playing.is_set()
                if (
                    cue_is_playing
                    and not self._pending_wake_confirmed
                    and not self._recorder.is_recording
                ):
                    # Our own cue is on the wire. Skipping these frames excises it from
                    # the utterance instead of recording Amy acknowledging herself.
                    continue
                if self.controller.should_drop_main_transcript():
                    self._suspend_capture()
                    continue
                self._capture_suspended = False

                vad_state = self._vad.observe(frame)
                if self._recorder.is_recording:
                    segment = self._recorder.feed(frame, vad_state)
                    if segment is not None:
                        self._enqueue_utterance(
                            UtteranceJob(
                                utterance_id=self._next_utterance_id(),
                                epoch=self.controller.epoch,
                                captured_at=time.monotonic(),
                                pcm=segment.pcm,
                                duration_seconds=segment.duration_seconds,
                                wake_confirmed=self._recording_wake_confirmed,
                            )
                        )
                        self._recording_wake_confirmed = False
                    continue

                if self._pending_wake_confirmed:
                    with self._pending_wake_lock:
                        snapshot = self._pending_wake_snapshot or self._ring_buffer.snapshot()
                        wake_confirmed = self._pending_wake_confirmed
                        self._pending_wake_confirmed = False
                        self._pending_wake_snapshot = None
                    self._recorder.begin(snapshot)
                    self._recording_wake_confirmed = wake_confirmed
                    segment = self._recorder.feed(frame, vad_state)
                    if segment is not None:
                        self._enqueue_utterance(
                            UtteranceJob(
                                utterance_id=self._next_utterance_id(),
                                epoch=self.controller.epoch,
                                captured_at=time.monotonic(),
                                pcm=segment.pcm,
                                duration_seconds=segment.duration_seconds,
                                wake_confirmed=wake_confirmed,
                            )
                        )
                        self._recording_wake_confirmed = False
                    continue

                if self.controller.get_status().active_conversation and vad_state.is_speech:
                    snapshot = self._ring_buffer.snapshot()
                    self._recorder.begin(snapshot)
                    self._recording_wake_confirmed = False
                    segment = self._recorder.feed(frame, vad_state)
                    if segment is not None:
                        self._enqueue_utterance(
                            UtteranceJob(
                                utterance_id=self._next_utterance_id(),
                                epoch=self.controller.epoch,
                                captured_at=time.monotonic(),
                                pcm=segment.pcm,
                                duration_seconds=segment.duration_seconds,
                            )
                        )
        except Exception as exc:  # pragma: no cover - runtime path
            self.controller.status.error_message = str(exc)
            self.on_status(f"capture error: {exc}")
        finally:
            self.on_status("pipeline loop exited")

    def _detect_wake_from_ring(self) -> bool:
        assert self._ring_buffer is not None
        assert self._wake_detector is not None
        snapshot = self._ring_buffer.snapshot(duration_ms=self.audio_config.wake_window_ms)
        if len(snapshot.frames) < self.audio_config.wake_min_window_frames:
            return False
        if snapshot.rms < self.audio_config.wake_rms_threshold:
            return False
        detection = self._wake_detector.detect(snapshot.pcm)
        if not detection.matched:
            return False
        with self._pending_wake_lock:
            self._pending_wake_snapshot = self._ring_buffer.snapshot()
            self._pending_wake_confirmed = True
        if detection.is_bare:
            self._greet_new_conversation()
        return True

    def _wake_loop(self) -> None:
        if self._wake_subscription is None:
            return
        assert self._wake_detector is not None
        assert self._wake_vad is not None
        assert self._ring_buffer is not None
        wake_subscription = self._wake_subscription
        config = self.audio_config
        wake_session = WakeSession(config)
        last_sequence = 0
        try:
            while not self._stop_event.is_set():
                frame = wake_subscription.receive(timeout=0.2)
                if frame is None:
                    if self._stop_event.is_set() or wake_subscription.is_closed:
                        break
                    continue
                if last_sequence and frame.sequence > last_sequence + 1:
                    # The gating counters below describe a continuous stream. Once
                    # frames go missing they no longer match the audio, so start over
                    # rather than polling on a window we cannot account for.
                    self._wake_gap_count += 1
                    logger.warning(
                        "wake loop lost %d frame(s) before sequence %d (%d total)",
                        frame.sequence - last_sequence - 1,
                        frame.sequence,
                        self._wake_gap_count,
                    )
                    self._wake_vad.reset()
                    wake_session.reset()
                last_sequence = frame.sequence
                status = self.controller.get_status()
                if (
                    status.active_conversation
                    or self.controller.should_drop_main_transcript()
                    or not self._capture_enabled.is_set()
                ):
                    self._wake_vad.reset()
                    self._wake_detector.reset()
                    wake_session.reset()
                    continue
                if self._recorder is not None and self._recorder.is_recording:
                    self._wake_vad.reset()
                    wake_session.reset()
                    continue

                is_speech = self._wake_vad.observe(frame).is_speech
                wake_session.observe(sequence=frame.sequence, is_speech=is_speech, energy=float(frame.rms))

                if wake_session.speech_active:
                    if wake_session.should_attempt():
                        wake_session.record_attempt(matched=self._detect_wake_from_ring())
                    continue

                if not wake_session.speech_finished:
                    continue
                if wake_session.should_attempt_trailing():
                    self._detect_wake_from_ring()
                self._wake_vad.reset()
                wake_session.reset()
        except Exception as exc:  # pragma: no cover - runtime path
            self.controller.status.error_message = str(exc)
            self.on_status(f"wake loop error: {exc}")
        finally:
            self.on_status("wake loop exited")

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
                wake_confirmed=job.wake_confirmed,
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
                self.controller.process_transcript(
                    job.transcript,
                    utterance_id=job.utterance_id,
                    wake_confirmed=job.wake_confirmed,
                )
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

    def _reset_audio_state(self) -> None:
        if self._ring_buffer is not None:
            self._ring_buffer.clear()
        self._reset_detection_state()

    def _suspend_capture(self) -> None:
        """Hold capture while Amy owns the channel.

        The ring buffer keeps filling through thinking and cooldown so a follow-up
        spoken the instant Amy finishes still has its pre-roll. It is only dropped
        while the speaker is live, where the only thing to capture is Amy's own voice.
        """
        if self._is_speaker_active() and self._ring_buffer is not None:
            self._ring_buffer.clear()
        if self._capture_suspended:
            return
        self._capture_suspended = True
        self._reset_detection_state()

    def wake_cue_available(self) -> bool:
        """True when the runtime will speak the greeting itself.

        Lets the controller skip its own slower `say` call without going silent if
        the output device turned out to be unavailable.
        """
        return self._cue_player is not None and self._cue_player.is_available

    def _greet_new_conversation(self) -> None:
        """Speak the greeting only when opening a conversation, not on every wake.

        Acknowledging here instead of after the recorder and the large transcription
        model saves well over a second. Once a conversation is live the user is being
        listened to already, so re-greeting mid-conversation is just noise; letting it
        go idle re-arms the greeting for the next one.
        """
        if self._cue_player is None:
            return
        if self.controller.get_status().active_conversation:
            return
        self._cue_player.play()

    def _reset_capture_state(self) -> None:
        """Drop the in-flight recording without discarding wake state.

        A fan-out gap only invalidates one consumer's timeline. The ring buffer sees
        every frame, so a wake already confirmed from ring audio is still sound.
        """
        if self._vad is not None:
            self._vad.reset()
        if self._recorder is not None:
            self._recorder.reset()
        self._recording_wake_confirmed = False

    def _reset_detection_state(self) -> None:
        self._reset_capture_state()
        if self._wake_vad is not None:
            self._wake_vad.reset()
        if self._wake_detector is not None:
            self._wake_detector.reset()
        with self._pending_wake_lock:
            self._pending_wake_confirmed = False
            self._pending_wake_snapshot = None

    def _is_speaker_active(self) -> bool:
        speaker_state = getattr(self.controller.speaker, "is_speaking", None)
        return bool(speaker_state.is_set()) if speaker_state is not None else False

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

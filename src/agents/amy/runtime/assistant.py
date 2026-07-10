from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import logging
import os
import queue
import subprocess
import threading
import time
from typing import Callable

from ..core.models import AssistantPhase
from ..modalities.audio import AudioConfig, MicrophoneSource, SpeechSegmenter, Transcriber
from ..core.controller import AssistantController


StatusCallback = Callable[[str], None]


logger = logging.getLogger(__name__)
command_logger = logging.getLogger("agents.command_listener")


@dataclass
class AssistantRuntime:
    controller: AssistantController
    transcriber: Transcriber
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    log_transcripts: bool = False
    on_status: StatusCallback = field(default=lambda _message: None)
    _transcript_queue: queue.Queue[str] = field(default_factory=lambda: queue.Queue[str](), init=False)
    _command_queue: queue.Queue[str] = field(default_factory=lambda: queue.Queue[str](), init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _capture_enabled: threading.Event = field(default_factory=threading.Event, init=False)
    _capture_thread: threading.Thread | None = field(default=None, init=False)
    _worker_thread: threading.Thread | None = field(default=None, init=False)
    _command_thread: threading.Thread | None = field(default=None, init=False)
    _acknowledgement_thread: threading.Thread | None = field(default=None, init=False)
    _acknowledgement_stop: threading.Event = field(default_factory=threading.Event, init=False)
    _acknowledgement_sound_path: str = field(init=False)
    _acknowledgement_process: subprocess.Popen[bytes] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._acknowledgement_sound_path = str(
            Path(__file__).resolve().parent / ".." / "assets" / "Glass.aiff"
        )

    def start(self) -> None:
        self._stop_event.clear()
        self._acknowledgement_stop.clear()
        self._capture_enabled.set()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._command_thread = threading.Thread(target=self._command_worker_loop, daemon=True)
        self._command_thread.start()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self.on_status("command listener active")
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

    def cut_channel(self) -> None:
        logger.debug("cut_channel requested")
        self.controller.cut_channel()
        self._capture_enabled.clear()
        self.on_status("channel cut")

    def stop(self) -> None:
        self._stop_event.set()
        self._acknowledgement_stop.set()
        self._stop_acknowledgement_process()
        self._capture_enabled.clear()
        self.controller.stop()
        self.on_status("runtime stopping")
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2)
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2)
        if self._command_thread is not None and self._command_thread.is_alive():
            self._command_thread.join(timeout=2)
        if self._acknowledgement_thread is not None and self._acknowledgement_thread.is_alive():
            self._acknowledgement_thread.join(timeout=2)

    def status_text(self) -> str:
        status = self.controller.get_status()
        return (
            f"phase={status.phase.value} "
            f"active={status.active_conversation} "
            f"paused={status.paused} "
            f"last_user={status.last_user_text!r} "
            f"last_assistant={status.last_assistant_text!r}"
        )

    def _capture_loop(self) -> None:
        speech_segmenter = SpeechSegmenter(self.audio_config)
        command_segmenter = SpeechSegmenter(self._command_audio_config())
        try:
            with MicrophoneSource(self.audio_config) as microphone:
                for frame in microphone.frames():
                    if self._stop_event.is_set() or not self._capture_enabled.is_set():
                        break
                    command_segment = command_segmenter.feed(frame)
                    if command_segment is not None:
                        if command_segment.duration_seconds > 2.5:
                            continue
                        command_text = self.transcriber.transcribe(command_segment.path)
                        if self.controller.is_interrupt_command(command_text):
                            self._command_queue.put(command_text)

                    if self.controller.should_drop_main_transcript():
                        speech_segmenter.reset()
                        continue

                    segment = speech_segmenter.feed(frame)
                    if segment is None:
                        continue
                    text = self.transcriber.transcribe(segment.path)
                    self._log_transcript("main", text)
                    if not self._should_queue_main_transcript():
                        continue
                    self._transcript_queue.put(text)
        except Exception as exc:  # pragma: no cover - runtime path
            self.controller.status.error_message = str(exc)
            self.on_status(f"capture error: {exc}")
        finally:
            self.on_status("capture loop exited")

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                transcript = self._transcript_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not transcript.strip():
                continue
            try:
                self.controller.process_transcript(transcript)
            except Exception as exc:  # pragma: no cover - runtime path
                self.controller.status.error_message = str(exc)
                self.on_status(f"worker error: {exc}")
            finally:
                time.sleep(0.01)

    def _command_worker_loop(self) -> None:
        command_logger.debug("worker started")
        while not self._stop_event.is_set():
            try:
                transcript = self._command_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not transcript.strip():
                continue
            try:
                command_logger.debug("processing transcript: %r", transcript)
                self._log_transcript("command", transcript)
                self.handle_command_transcript(transcript)
            except Exception as exc:  # pragma: no cover - runtime path
                self.controller.status.error_message = str(exc)
                self.on_status(f"command worker error: {exc}")
            finally:
                continue

    def _should_queue_main_transcript(self) -> bool:
        speaker_state = getattr(self.controller.speaker, "is_speaking", None)
        is_speaking = bool(speaker_state.is_set()) if speaker_state is not None else False
        if is_speaking:
            return False

        if self.controller.should_drop_main_transcript():
            return False

        status = self.controller.get_status()
        return status.phase not in {AssistantPhase.PAUSED}

    def _log_transcript(
        self,
        source: str,
        transcript: str,
    ) -> None:
        if not self.log_transcripts:
            return
        logger.info("transcribed %s transcript: %r", source, transcript)

    def handle_command_transcript(self, transcript: str) -> None:
        if self.controller.is_interrupt_command(transcript):
            command_logger.debug("matched interrupt: %r", transcript)
            self.controller.process_transcript(transcript)

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

    def _command_audio_config(self) -> AudioConfig:
        return replace(
            self.audio_config,
            frame_ms=max(20, min(self.audio_config.frame_ms, 30)),
            pre_roll_ms=max(120, self.audio_config.pre_roll_ms // 2),
            silence_ms=max(180, self.audio_config.silence_ms // 2),
        )

__all__ = ["AssistantRuntime"]

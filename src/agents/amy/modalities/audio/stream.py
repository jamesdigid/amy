from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
import queue
import threading
from typing import TypeAlias

from .capture import MicrophoneSource
from .models import AudioConfig, AudioFrame

logger = logging.getLogger(__name__)

_QueueItem: TypeAlias = AudioFrame | None


def _empty_observer_list() -> list[Callable[[AudioFrame], None]]:
    return []


def _empty_subscription_list() -> list[FrameSubscription]:
    return []


@dataclass
class FrameSubscription:
    name: str
    maxsize: int
    _queue: queue.Queue[_QueueItem] = field(init=False, repr=False)
    dropped_frames: int = field(default=0, init=False)
    _closed: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def __post_init__(self) -> None:
        self._queue = queue.Queue(maxsize=self.maxsize)

    def put(self, frame: AudioFrame) -> None:
        if self._closed.is_set():
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(frame)
            self.dropped_frames += 1

    def receive(self, timeout: float | None = None) -> AudioFrame | None:
        if self._closed.is_set() and self._queue.empty():
            return None
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is None:
            self._closed.set()
            return None
        return item

    @property
    def is_closed(self) -> bool:
        """True once the publisher has stopped and every buffered frame is drained."""
        return self._closed.is_set() and self._queue.empty()

    def close(self) -> None:
        self._closed.set()


@dataclass
class AudioStream:
    config: AudioConfig
    microphone_factory: Callable[[AudioConfig], MicrophoneSource] = MicrophoneSource
    on_error: Callable[[Exception], None] | None = None
    _observers: list[Callable[[AudioFrame], None]] = field(
        default_factory=_empty_observer_list,
        init=False,
        repr=False,
    )
    _subscriptions: list[FrameSubscription] = field(
        default_factory=_empty_subscription_list,
        init=False,
        repr=False,
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _sequence: int = field(default=0, init=False, repr=False)

    def add_observer(self, observer: Callable[[AudioFrame], None]) -> None:
        with self._lock:
            self._observers.append(observer)

    def subscribe(self, name: str, maxsize: int = 8) -> FrameSubscription:
        subscription = FrameSubscription(name=name, maxsize=maxsize)
        with self._lock:
            self._subscriptions.append(subscription)
        return subscription

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            subscriptions = list(self._subscriptions)
            thread = self._thread
        for subscription in subscriptions:
            subscription.close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)

    def _run(self) -> None:
        try:
            with self.microphone_factory(self.config) as microphone:
                for frame in microphone.frames():
                    if self._stop_event.is_set():
                        break
                    self._sequence += 1
                    sequenced_frame = AudioFrame(
                        data=frame.data,
                        overflow=frame.overflow,
                        sequence=self._sequence,
                    )
                    with self._lock:
                        observers = tuple(self._observers)
                        subscriptions = tuple(self._subscriptions)
                    for observer in observers:
                        observer(sequenced_frame)
                    for subscription in subscriptions:
                        subscription.put(sequenced_frame)
        except Exception as exc:
            logger.exception("audio stream stopped unexpectedly")
            if self.on_error is not None:
                self.on_error(exc)
        finally:
            with self._lock:
                subscriptions = tuple(self._subscriptions)
            for subscription in subscriptions:
                subscription.close()


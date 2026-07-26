from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, cast

from .models import AudioConfig, AudioFrame


class _InputStream(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def read(self, frames: int) -> tuple[bytes, bool]: ...


@dataclass
class MicrophoneSource:
    config: AudioConfig
    _stream: _InputStream | None = None

    def __enter__(self) -> "MicrophoneSource":
        import sounddevice as sd

        stream = cast(
            _InputStream,
            sd.RawInputStream(
                device=self.config.input_device,
                channels=1,
                samplerate=self.config.sample_rate,
                dtype="int16",
                blocksize=self.config.frame_samples,
            ),
        )
        self._stream = stream
        stream.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.abort()

    def abort(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def frames(self) -> Iterator[AudioFrame]:
        if self._stream is None:
            raise RuntimeError("MicrophoneSource must be entered before reading frames")

        while True:
            data, overflow = self._stream.read(self.config.frame_samples)
            yield AudioFrame(data=bytes(data), overflow=overflow)

__all__ = ["MicrophoneSource"]

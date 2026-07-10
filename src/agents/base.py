from __future__ import annotations

from pathlib import Path
from typing import Protocol


class RunnableAgent(Protocol):
    def run(self) -> int: ...


class AgentBuilder(Protocol):
    def build(self, workspace: Path | None = None) -> RunnableAgent: ...

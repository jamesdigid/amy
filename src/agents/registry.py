from __future__ import annotations

from pathlib import Path

from .base import RunnableAgent


def available_agents() -> tuple[str, ...]:
    return ("amy",)


def build_agent(name: str, workspace: Path | None = None) -> RunnableAgent:
    if name == "amy":
        from .amy.agent import AmyAgent

        return AmyAgent.build(workspace)
    raise ValueError(f"Unknown agent: {name}")

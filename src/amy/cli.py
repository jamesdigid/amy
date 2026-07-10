from __future__ import annotations

from agents.amy.agent import AmyAgent


def main() -> int:
    return AmyAgent.build().run()

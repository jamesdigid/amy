from __future__ import annotations

import sys
from pathlib import Path

from .registry import available_agents, build_agent


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    agent_name = args[0] if args else "amy"
    if agent_name in {"-h", "--help"}:
        print("Usage: python -m agents [agent-name]")
        print("Available agents:", ", ".join(available_agents()))
        return 0
    agent = build_agent(agent_name, workspace=Path.cwd())
    return agent.run()

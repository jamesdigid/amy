from __future__ import annotations

from dataclasses import dataclass
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .registry import available_agents, build_agent


@dataclass(frozen=True)
class ServicePaths:
    workspace: Path
    venv_dir: Path
    pid_file: Path
    log_file: Path
    setup_marker: Path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    command = args.command or "run"
    workspace = Path.cwd()
    paths = _service_paths(workspace)

    try:
        if command == "run":
            return _run_agent(args.agent, workspace, interactive=not args.non_interactive)
        if command == "setup":
            _setup_environment(paths)
            return 0
        if command in {"start", "deploy"}:
            return _start_agent(paths, args.agent)
        if command == "stop":
            return _stop_agent(paths)
        if command == "status":
            return _status_agent(paths)
    except ValueError as exc:
        if str(exc) == "OPENAI_API_KEY is required":
            print("OPENAI_API_KEY is required to run Amy. Export it and rerun the command.")
            return 1
        raise

    parser.error(f"Unknown command: {command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amy", description="Amy assistant lifecycle commands")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the assistant in the foreground")
    run_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without the terminal command loop",
    )
    run_parser.add_argument(
        "agent",
        nargs="?",
        default="amy",
        choices=available_agents(),
        help="Agent to run",
    )
    run_parser.set_defaults(command="run")

    setup_parser = subparsers.add_parser("setup", help="Create the local venv and install deps")
    setup_parser.add_argument(
        "agent",
        nargs="?",
        default="amy",
        choices=available_agents(),
        help="Agent to prepare",
    )
    setup_parser.set_defaults(command="setup")

    start_parser = subparsers.add_parser("start", help="Start the assistant in the background")
    start_parser.add_argument(
        "agent",
        nargs="?",
        default="amy",
        choices=available_agents(),
        help="Agent to start",
    )
    start_parser.set_defaults(command="start")

    stop_parser = subparsers.add_parser("stop", help="Stop the background assistant")
    stop_parser.add_argument(
        "agent",
        nargs="?",
        default="amy",
        choices=available_agents(),
        help="Agent to stop",
    )
    stop_parser.set_defaults(command="stop")

    status_parser = subparsers.add_parser("status", help="Show the background assistant status")
    status_parser.add_argument(
        "agent",
        nargs="?",
        default="amy",
        choices=available_agents(),
        help="Agent to inspect",
    )
    status_parser.set_defaults(command="status")

    deploy_parser = subparsers.add_parser("deploy", help="Set up and start the assistant")
    deploy_parser.add_argument(
        "agent",
        nargs="?",
        default="amy",
        choices=available_agents(),
        help="Agent to deploy",
    )
    deploy_parser.set_defaults(command="deploy")

    parser.set_defaults(agent="amy", non_interactive=False)
    return parser


def _run_agent(agent_name: str, workspace: Path, *, interactive: bool) -> int:
    agent = build_agent(agent_name, workspace=workspace)
    return agent.run(interactive=interactive)


def _service_paths(workspace: Path) -> ServicePaths:
    return ServicePaths(
        workspace=workspace,
        venv_dir=workspace / ".venv",
        pid_file=workspace / ".amy" / "amy.pid",
        log_file=workspace / ".amy" / "amy.log",
        setup_marker=workspace / ".amy" / "setup.complete",
    )


def _setup_environment(paths: ServicePaths) -> None:
    subprocess.run([str(paths.workspace / "scripts" / "amy"), "setup"], cwd=paths.workspace, check=True)


def _ensure_environment(paths: ServicePaths) -> None:
    venv_python = _venv_python(paths)
    if paths.setup_marker.exists() and venv_python.exists():
        return
    _setup_environment(paths)


def _start_agent(paths: ServicePaths, agent_name: str) -> int:
    _ensure_environment(paths)
    running_pid = _read_pid(paths)
    if running_pid is not None:
        print(f"Amy is already running (pid {running_pid}).")
        return 0

    paths.pid_file.parent.mkdir(parents=True, exist_ok=True)
    paths.log_file.parent.mkdir(parents=True, exist_ok=True)

    venv_python = _venv_python(paths)
    with paths.log_file.open("ab") as log_file:
        process = subprocess.Popen(
            [str(venv_python), "-m", "agents", "run", "--non-interactive", agent_name],
            cwd=paths.workspace,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )

    paths.pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    time.sleep(0.5)
    if process.poll() is not None:
        _cleanup_pidfile(paths)
        print(f"Amy failed to start. Check {paths.log_file} for details.")
        return process.returncode or 1

    print(f"Amy started in the background (pid {process.pid}).")
    print(f"Logs: {paths.log_file}")
    return 0


def _stop_agent(paths: ServicePaths) -> int:
    pid = _read_pid(paths)
    if pid is None:
        print("Amy is not running.")
        return 0

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _cleanup_pidfile(paths)
        print("Amy was not running.")
        return 0

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not _process_is_running(pid):
            _cleanup_pidfile(paths)
            print("Amy stopped.")
            return 0
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _cleanup_pidfile(paths)
    print("Amy stop timed out; sent SIGKILL.")
    return 0


def _status_agent(paths: ServicePaths) -> int:
    pid = _read_pid(paths)
    if pid is None:
        print("Amy is not running.")
        return 0

    print(f"Amy is running (pid {pid}).")
    print(f"Logs: {paths.log_file}")
    return 0


def _read_pid(paths: ServicePaths) -> int | None:
    if not paths.pid_file.exists():
        return None

    try:
        pid = int(paths.pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        _cleanup_pidfile(paths)
        return None

    if _process_is_running(pid):
        return pid

    _cleanup_pidfile(paths)
    return None


def _cleanup_pidfile(paths: ServicePaths) -> None:
    try:
        paths.pid_file.unlink()
    except FileNotFoundError:
        pass


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _venv_python(paths: ServicePaths) -> Path:
    if os.name == "nt":
        return paths.venv_dir / "Scripts" / "python.exe"
    return paths.venv_dir / "bin" / "python"

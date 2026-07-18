from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents import cli


class CliTests(unittest.TestCase):
    def test_main_defaults_to_interactive_run(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self.run_calls: list[bool] = []

            def run(self, *, interactive: bool = True) -> int:
                self.run_calls.append(interactive)
                return 0

        fake_agent = FakeAgent()

        with patch.object(cli, "build_agent", return_value=fake_agent):
            exit_code = cli.main([])

        self.assertEqual(exit_code, 0)
        self.assertEqual(fake_agent.run_calls, [True])

    def test_setup_delegates_to_bootstrap_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            paths = cli._service_paths(workspace)
            scripts_dir = workspace / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            (scripts_dir / "amy").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

            with patch.object(cli.subprocess, "run") as run_mock:
                cli._setup_environment(paths)

        run_mock.assert_called_once()
        command = run_mock.call_args.args[0]
        self.assertEqual(command, [str(workspace / "scripts" / "amy"), "setup"])
        self.assertEqual(run_mock.call_args.kwargs["cwd"], workspace)

    def test_start_writes_pid_and_uses_background_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            paths = cli._service_paths(workspace)
            paths.setup_marker.parent.mkdir(parents=True, exist_ok=True)
            paths.setup_marker.write_text("ready\n", encoding="utf-8")
            venv_python = workspace / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("", encoding="utf-8")

            process = MagicMock()
            process.pid = 4321
            process.poll.return_value = None

            with (
                patch.object(cli, "_ensure_environment") as ensure_environment,
                patch.object(cli, "_venv_python", return_value=venv_python),
                patch.object(cli.subprocess, "Popen", return_value=process) as popen,
                patch.object(cli.time, "sleep", return_value=None),
            ):
                exit_code = cli._start_agent(paths, "amy")

            self.assertEqual(exit_code, 0)
            ensure_environment.assert_called_once_with(paths)
            popen.assert_called_once()
            self.assertIn("--non-interactive", popen.call_args.args[0])
            self.assertTrue(paths.pid_file.exists())
            self.assertEqual(paths.pid_file.read_text(encoding="utf-8").strip(), "4321")

    def test_read_pid_removes_stale_pidfile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            paths = cli._service_paths(workspace)
            paths.pid_file.parent.mkdir(parents=True, exist_ok=True)
            paths.pid_file.write_text("9999\n", encoding="utf-8")

            with patch.object(cli, "_process_is_running", return_value=False):
                pid = cli._read_pid(paths)

            self.assertIsNone(pid)
            self.assertFalse(paths.pid_file.exists())

    def test_status_returns_success_when_not_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = cli._service_paths(Path(temp_dir))

            exit_code = cli._status_agent(paths)

            self.assertEqual(exit_code, 0)

    def test_main_reports_missing_openai_key_cleanly(self) -> None:
        with patch.object(cli, "build_agent", side_effect=ValueError("OPENAI_API_KEY is required")):
            with patch("builtins.print") as print_mock:
                exit_code = cli.main(["run", "amy"])

        self.assertEqual(exit_code, 1)
        print_mock.assert_called_once_with(
            "OPENAI_API_KEY is required to run Amy. Export it and rerun the command."
        )

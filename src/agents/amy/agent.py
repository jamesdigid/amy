from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import os

from ..providers.openai import OpenAIResponder
from .config import AppConfig, load_config
from .core.controller import AssistantController
from .core.prompts import PromptBuilder
from .memory import MemoryStore, OpenAIMemoryClassifier
from .modalities.audio import AudioConfig, FasterWhisperTranscriber, LocalSpeaker
from .runtime.assistant import AssistantRuntime
from .runtime.status import AmyStatusReporter
from .skills.registry import AmySkillRegistry
from .skills.browser import DuckDuckGoWebSearch


@dataclass
class AmyAgent:
    config: AppConfig
    runtime: AssistantRuntime

    @classmethod
    def build(cls, workspace: Path | None = None) -> "AmyAgent":
        _configure_logging()
        config = load_config(workspace)
        runtime_holder: dict[str, AssistantRuntime] = {}
        prompt_builder = PromptBuilder(
            assistant_name=config.assistant_name,
            project_context=config.project_context,
            wake_word=config.wake_word,
            recent_turns=config.recent_turns,
        )
        memory_store = MemoryStore(memory_dir=config.memory_dir)
        memory_classifier = OpenAIMemoryClassifier(api_key=config.api_key, model=config.model)
        responder = OpenAIResponder(api_key=config.api_key, model=config.model)
        speaker = LocalSpeaker()
        skill_registry = AmySkillRegistry(
            project_root=workspace or Path.cwd(),
            memory_store=memory_store,
            web_search=DuckDuckGoWebSearch(),
        )
        status_reporter = AmyStatusReporter(
            memory_dir=config.memory_dir,
            skill_registry=skill_registry,
            web_search_enabled=True,
            transcript_logging_enabled=config.log_transcripts,
        )

        def runtime_acknowledgement_start() -> None:
            runtime = runtime_holder.get("runtime")
            if runtime is not None:
                runtime.play_acknowledgement_loop()

        def runtime_acknowledgement_stop() -> None:
            runtime = runtime_holder.get("runtime")
            if runtime is not None:
                runtime.stop_acknowledgement_loop()

        controller = AssistantController(
            prompt_builder=prompt_builder,
            responder=responder,
            speaker=speaker,
            wake_word=config.wake_word,
            status_reporter=status_reporter,
            memory_store=memory_store,
            memory_classifier=memory_classifier,
            web_search=skill_registry.web_search,
            acknowledgment_callback=runtime_acknowledgement_start,
            acknowledgment_stop_callback=runtime_acknowledgement_stop,
            usage_logger=lambda tokens, cost: print(
                f"[amy] estimated usage: {tokens} tokens (~${cost:.4f})"
            ),
        )
        runtime = AssistantRuntime(
            controller=controller,
            transcriber=FasterWhisperTranscriber(language=config.transcript_language),
            audio_config=AudioConfig(),
            log_transcripts=config.log_transcripts,
            status_reporter=status_reporter,
            on_status=lambda message: print(f"[amy] {message}"),
        )
        runtime_holder["runtime"] = runtime
        return cls(config=config, runtime=runtime)

    def run(self) -> int:
        print("Amy is ready.")
        print("Commands: pause, resume, status, quit")
        self.runtime.start()
        try:
            while True:
                command = input("> ").strip().lower()
                if command in {"quit", "exit"}:
                    break
                if command == "pause":
                    self.runtime.pause_capture()
                    continue
                if command == "resume":
                    self.runtime.resume_capture()
                    continue
                if command == "status":
                    print(self.runtime.status_text())
                    continue
                if command:
                    print("Unknown command. Use pause, resume, status, or quit.")
        except KeyboardInterrupt:
            print("\nShutting down.")
        finally:
            self.runtime.stop()
        return 0


def _configure_logging() -> None:
    main_log_level_name = os.environ.get("AMY_MAIN_LOG_LEVEL", "DEBUG").upper()
    main_log_level = getattr(logging, main_log_level_name, logging.DEBUG)
    command_log_level_name = os.environ.get("AMY_COMMAND_LOG_LEVEL", "WARNING").upper()
    command_log_level = getattr(logging, command_log_level_name, logging.WARNING)
    logging.basicConfig(level=main_log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("agents.amy.core.controller").setLevel(main_log_level)
    logging.getLogger("agents.amy.runtime").setLevel(main_log_level)
    logging.getLogger("amy.command_listener").setLevel(command_log_level)

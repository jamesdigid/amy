from __future__ import annotations

import threading
import time
import tempfile
import unittest
from pathlib import Path

from agents.amy.core.controller import AssistantController
from agents.amy.core.models import Message
from agents.amy.core.prompts import PromptBuilder
from agents.amy.memory import (
    MemoryClassifierProtocol,
    MemoryDecision,
    MemoryDraft,
    MemoryStore,
    MemoryStoreProtocol,
)
from agents.amy.runtime.status import AmyStatusReporter
from agents.amy.skills.browser import SearchResult


class FakeResponder:
    def __init__(self, response: str = "reply") -> None:
        self.calls: list[list[Message]] = []
        self.response = response

    def generate_reply(self, messages: list[Message], cancel_event: threading.Event) -> str:
        self.calls.append(messages)
        if cancel_event.is_set():
            return ""
        return self.response


class FakeSpeaker:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.stopped = 0

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def stop(self) -> None:
        self.stopped += 1


class FakeAcknowledgementLoop:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


class StopBeforeSpeakSpeaker(FakeSpeaker):
    def __init__(self, ack_loop: FakeAcknowledgementLoop) -> None:
        super().__init__()
        self._ack_loop = ack_loop
        self.saw_stopped = False

    def speak(self, text: str) -> None:
        if text != "Amy here":
            self.saw_stopped = self._ack_loop.stopped >= 1
        super().speak(text)


class FakeWebSearch:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, limit: int = 4) -> list[SearchResult]:
        self.queries.append((query, limit))
        return [
            SearchResult(
                title="Example result",
                url="https://example.com",
                snippet="Example snippet",
                content="Example article text with details.",
            )
        ]


class FakeMemoryStore:
    def __init__(self, response: str = "### Memory: team.md\nTeam memory") -> None:
        self.response = response
        self.prompts: list[str] = []
        self.saved: list[MemoryDraft] = []

    def retrieve_context(self, prompt: str, limit: int = 3) -> str:
        self.prompts.append(prompt)
        return self.response

    def draft_from_prompt(self, prompt: str, subject: str | None = None) -> MemoryDraft | None:
        return MemoryDraft(
            path=Path("/tmp/team.md"),
            tags=("team",),
            summary=prompt,
            memories=(prompt,),
            retrieval_notes=("draft",),
            content="# Memory\n",
        )

    def save_draft(self, draft: MemoryDraft) -> Path:
        self.saved.append(draft)
        return draft.path


class FakeMemoryClassifier:
    def __init__(self, should_save: bool = False, subject: str = "") -> None:
        self.should_save = should_save
        self.subject = subject
        self.calls: list[str] = []

    def classify(self, prompt: str, cancel_event: threading.Event) -> MemoryDecision:
        self.calls.append(prompt)
        return MemoryDecision(
            should_save=self.should_save,
            subject=self.subject,
            confidence=0.99 if self.should_save else 0.01,
            reason="test classifier",
        )


def build_controller(
    web_search: FakeWebSearch | None = None,
    memory_store: MemoryStoreProtocol | None = None,
    memory_classifier: MemoryClassifierProtocol | None = None,
    status_reporter: AmyStatusReporter | None = None,
    idle_timeout_seconds: float = 0.1,
    speech_cooldown_seconds: float = 0.05,
    follow_up_timeout_seconds: float = 0.1,
) -> tuple[AssistantController, FakeResponder, FakeSpeaker, FakeWebSearch | None]:
    responder = FakeResponder()
    speaker = FakeSpeaker()
    controller = AssistantController(
        prompt_builder=PromptBuilder(
                assistant_name="Amy",
                project_context="",
                wake_word="amy",
        ),
        responder=responder,
        speaker=speaker,
        wake_word="amy",
        status_reporter=status_reporter,
        memory_store=memory_store,
        memory_classifier=memory_classifier,
        idle_timeout_seconds=idle_timeout_seconds,
        web_search=web_search,
    )
    controller.speech_cooldown_seconds = speech_cooldown_seconds
    controller.follow_up_timeout_seconds = follow_up_timeout_seconds
    return controller, responder, speaker, web_search


class AssistantControllerTests(unittest.TestCase):
    def test_requires_wake_word(self) -> None:
        controller, responder, speaker, _ = build_controller()

        result = controller.process_transcript("hello there")

        self.assertIsNone(result)
        self.assertEqual(responder.calls, [])
        self.assertEqual(speaker.spoken, [])

    def test_status_check_skips_responder_and_speaks_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status_reporter = AmyStatusReporter(
                memory_dir=Path(temp_dir),
                web_search_enabled=False,
            )
            controller, responder, speaker, _ = build_controller(status_reporter=status_reporter)

            result = controller.process_transcript("amy status check")

            self.assertIsNotNone(result)
            self.assertIn("Status check:", result or "")
            self.assertEqual(responder.calls, [])
            self.assertEqual(speaker.spoken, [result or ""])

    def test_handles_wake_and_reply(self) -> None:
        controller, responder, speaker, _ = build_controller()

        result = controller.process_transcript("amy summarize this")

        self.assertEqual(result, "reply")
        self.assertEqual(responder.calls[0][-1].content, "summarize this")
        self.assertEqual(speaker.spoken, ["reply"])
        self.assertTrue(controller.get_status().active_conversation)
        self.assertEqual(controller.get_status().phase.value, "cooldown")

        time.sleep(0.08)

        self.assertEqual(controller.get_status().phase.value, "listening")

        time.sleep(0.2)

        self.assertFalse(controller.get_status().active_conversation)
        self.assertEqual(controller.get_status().phase.value, "idle")

    def test_wake_word_alone_only_acknowledges(self) -> None:
        controller, responder, speaker, _ = build_controller()

        result = controller.process_transcript("amy")

        self.assertIsNone(result)
        self.assertEqual(responder.calls, [])
        self.assertEqual(speaker.spoken, ["Amy here"])
        self.assertTrue(controller.get_status().active_conversation)
        self.assertEqual(controller.get_status().phase.value, "cooldown")

        time.sleep(0.08)

        self.assertEqual(controller.get_status().phase.value, "awaiting_user_response")
        self.assertTrue(controller.get_status().active_conversation)

        time.sleep(0.15)

        self.assertFalse(controller.get_status().active_conversation)
        self.assertEqual(controller.get_status().phase.value, "idle")

    def test_wake_word_followed_by_natural_speech_becomes_query(self) -> None:
        controller, responder, speaker, _ = build_controller()

        result = controller.process_transcript("Amy, can you tell me the news in West Palm today")

        self.assertIsNotNone(result)
        self.assertEqual(responder.calls[0][-1].content, "can you tell me the news in West Palm today")
        self.assertEqual(speaker.spoken, ["reply"])
        self.assertTrue(controller.get_status().active_conversation)
        self.assertEqual(controller.get_status().phase.value, "cooldown")

        time.sleep(0.08)

        self.assertEqual(controller.get_status().phase.value, "listening")

        time.sleep(0.2)

        self.assertFalse(controller.get_status().active_conversation)
        self.assertEqual(controller.get_status().phase.value, "idle")

    def test_acknowledgement_echo_is_ignored(self) -> None:
        controller, responder, speaker, _ = build_controller()

        result = controller.process_transcript("Amy here")

        self.assertIsNone(result)
        self.assertEqual(responder.calls, [])
        self.assertEqual(speaker.spoken, [])

    def test_short_assistant_echo_is_ignored(self) -> None:
        responder = FakeResponder(response="Yes.")
        speaker = FakeSpeaker()
        controller = AssistantController(
            prompt_builder=PromptBuilder(
                assistant_name="Amy",
                project_context="",
                wake_word="amy",
            ),
            responder=responder,
            speaker=speaker,
            wake_word="amy",
        )

        first_result = controller.process_transcript("amy are you still there")
        second_result = controller.process_transcript("Yes!")

        self.assertEqual(first_result, "Yes.")
        self.assertIsNone(second_result)
        self.assertEqual(len(responder.calls), 1)
        self.assertEqual(speaker.spoken, ["Yes."])

    def test_memory_request_saves_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_store = MemoryStore(memory_dir=Path(temp_dir))
            memory_classifier = FakeMemoryClassifier(should_save=True, subject="favorite editor is vim")
            controller, responder, speaker, _ = build_controller(
                memory_store=memory_store,
                memory_classifier=memory_classifier,
            )

            reply = controller.process_transcript("amy my favorite editor is vim")

            self.assertIn("favorite.editor.vim.md", reply or "")
            self.assertEqual(responder.calls, [])
            self.assertEqual(speaker.spoken, ["Saved as `favorite.editor.vim.md`."])
            self.assertEqual(memory_classifier.calls, ["my favorite editor is vim"])
            saved_path = Path(temp_dir) / "favorite.editor.vim.md"
            self.assertTrue(saved_path.exists())
            self.assertIn("favorite editor is vim", saved_path.read_text(encoding="utf-8").lower())

    def test_memory_request_falls_back_when_classifier_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_store = MemoryStore(memory_dir=Path(temp_dir))
            controller, responder, speaker, _ = build_controller(memory_store=memory_store)

            reply = controller.process_transcript("amy remember that sky is blue")

            self.assertIn("sky.blue.md", reply or "")
            self.assertEqual(responder.calls, [])
            self.assertEqual(speaker.spoken, ["Saved as `sky.blue.md`."])

    def test_explicit_memory_request_saves_even_if_classifier_declines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_store = MemoryStore(memory_dir=Path(temp_dir))
            memory_classifier = FakeMemoryClassifier(should_save=False)
            controller, responder, speaker, _ = build_controller(
                memory_store=memory_store,
                memory_classifier=memory_classifier,
            )

            reply = controller.process_transcript("amy remember that sky is blue")

            self.assertIn("sky.blue.md", reply or "")
            self.assertEqual(memory_classifier.calls, ["remember that sky is blue"])
            self.assertEqual(responder.calls, [])
            self.assertEqual(speaker.spoken, ["Saved as `sky.blue.md`."])

    def test_acknowledgement_prefix_is_not_ignored(self) -> None:
        controller, responder, speaker, _ = build_controller()

        result = controller.process_transcript("Amy here tell me a story")

        self.assertEqual(result, "reply")
        self.assertEqual(responder.calls[0][-1].content, "tell me a story")
        self.assertEqual(speaker.spoken, ["reply"])

    def test_multiple_follow_up_questions_are_trimmed_to_one(self) -> None:
        controller, responder, speaker, _ = build_controller()
        responder.response = "Please share the purpose of the email? What tone should it have? Who is the recipient?"

        result = controller.process_transcript("amy draft an email")

        self.assertEqual(result, "Please share the purpose of the email?")
        self.assertEqual(speaker.spoken, ["Please share the purpose of the email?"])
        self.assertEqual(responder.calls[0][-1].content, "draft an email")
        self.assertEqual(controller.get_status().phase.value, "cooldown")

    def test_open_ended_question_expires_after_grace_period(self) -> None:
        responder = FakeResponder(response="What would you like me to do next?")
        speaker = FakeSpeaker()
        controller = AssistantController(
            prompt_builder=PromptBuilder(
                assistant_name="Amy",
                project_context="",
                wake_word="amy",
            ),
            responder=responder,
            speaker=speaker,
            wake_word="amy",
            idle_timeout_seconds=0.05,
        )
        controller.speech_cooldown_seconds = 0.05
        controller.follow_up_timeout_seconds = 0.2

        result = controller.process_transcript("amy summarize this")

        self.assertEqual(result, "What would you like me to do next?")
        self.assertEqual(controller.get_status().phase.value, "cooldown")

        time.sleep(0.08)

        self.assertEqual(controller.get_status().phase.value, "awaiting_user_response")

        follow_up = controller.process_transcript("tell me more")

        self.assertEqual(follow_up, "What would you like me to do next?")
        self.assertEqual(len(responder.calls), 2)
        self.assertEqual(responder.calls[-1][-1].content, "tell me more")
        self.assertEqual(controller.get_status().phase.value, "cooldown")

        time.sleep(0.08)

        self.assertEqual(controller.get_status().phase.value, "awaiting_user_response")

        time.sleep(0.25)

        self.assertFalse(controller.get_status().active_conversation)
        self.assertEqual(controller.get_status().phase.value, "idle")

    def test_wrap_up_question_keeps_session_open_for_follow_up(self) -> None:
        responder = FakeResponder(response="Is there anything else I can help you with?")
        speaker = FakeSpeaker()
        controller = AssistantController(
            prompt_builder=PromptBuilder(
                assistant_name="Amy",
                project_context="",
                wake_word="amy",
            ),
            responder=responder,
            speaker=speaker,
            wake_word="amy",
            idle_timeout_seconds=0.05,
        )
        controller.speech_cooldown_seconds = 0.05
        controller.follow_up_timeout_seconds = 0.2

        result = controller.process_transcript("amy summarize this")

        self.assertEqual(result, "Is there anything else I can help you with?")
        self.assertEqual(speaker.spoken, ["Is there anything else I can help you with?"])
        self.assertTrue(controller.get_status().active_conversation)
        self.assertEqual(controller.get_status().phase.value, "cooldown")

        time.sleep(0.08)

        self.assertEqual(controller.get_status().phase.value, "awaiting_user_response")
        self.assertTrue(controller.get_status().active_conversation)

        time.sleep(0.25)

        self.assertFalse(controller.get_status().active_conversation)
        self.assertEqual(controller.get_status().phase.value, "idle")

    def test_pause_and_cut_commands_change_state(self) -> None:
        controller, _responder, speaker, _ = build_controller()

        controller.process_transcript("amy start")
        controller.process_transcript("pause conversation")

        self.assertTrue(controller.get_status().active_conversation)
        self.assertTrue(controller.get_status().paused)
        self.assertEqual(controller.get_status().phase.value, "paused")
        self.assertGreaterEqual(speaker.stopped, 1)

    def test_resume_continues_preserved_conversation(self) -> None:
        controller, responder, _speaker, _ = build_controller()

        controller.process_transcript("amy start")
        controller.pause()
        self.assertTrue(controller.get_status().paused)
        self.assertEqual(controller.get_status().phase.value, "paused")
        controller.resume()
        self.assertFalse(controller.get_status().paused)
        self.assertEqual(controller.get_status().phase.value, "awaiting_user_response")
        controller.process_transcript("amy continue this thread")

        self.assertGreaterEqual(len(responder.calls), 2)
        self.assertEqual(responder.calls[-1][-1].content, "continue this thread")
        self.assertEqual(responder.calls[-1][1].content, "start")

    def test_pause_blocks_transcripts_until_resume(self) -> None:
        controller, responder, speaker, _ = build_controller()

        controller.process_transcript("amy summarize this")
        controller.pause()
        self.assertTrue(controller.get_status().paused)
        self.assertEqual(controller.get_status().phase.value, "paused")

        self.assertIsNone(controller.process_transcript("amy redirect to web search"))
        self.assertEqual(len(responder.calls), 1)

        controller.resume()
        self.assertFalse(controller.get_status().paused)
        self.assertEqual(controller.get_status().phase.value, "awaiting_user_response")

        controller.process_transcript("amy redirect to web search")

        self.assertGreaterEqual(speaker.stopped, 1)
        self.assertTrue(controller.get_status().active_conversation)
        self.assertEqual(responder.calls[-1][-1].content, "redirect to web search")

        time.sleep(0.2)

        self.assertFalse(controller.get_status().active_conversation)

    def test_interrupt_commands_are_detectable_during_speech(self) -> None:
        controller, _responder, _speaker, _ = build_controller()

        self.assertTrue(controller.is_interrupt_command("pause"))
        self.assertTrue(controller.is_interrupt_command("Amy pause"))
        self.assertTrue(controller.is_interrupt_command("resume"))
        self.assertTrue(controller.is_interrupt_command("cut channel"))
        self.assertTrue(controller.is_interrupt_command("stop"))
        self.assertFalse(controller.is_interrupt_command("would you like recent headlines, whether alerts or event pause"))
        self.assertFalse(controller.is_interrupt_command("hello there"))

    def test_search_prompt_injects_web_context(self) -> None:
        web_search = FakeWebSearch()
        controller, responder, _speaker, web_search = build_controller(web_search)

        controller.process_transcript("amy search web for python dataclasses")

        assert web_search is not None
        self.assertEqual(web_search.queries, [("python dataclasses", 4)])
        self.assertIn("Example result", responder.calls[0][0].content)
        self.assertIn("Example article text with details.", responder.calls[0][0].content)
        self.assertNotIn("https://example.com", responder.calls[0][0].content)

    def test_prompt_injects_relevant_memory_context(self) -> None:
        memory_store = FakeMemoryStore()
        controller, responder, _speaker, _ = build_controller(memory_store=memory_store)

        controller.process_transcript("amy remind me about the team memory")

        self.assertEqual(memory_store.prompts, ["remind me about the team memory"])
        self.assertIn("Relevant memories", responder.calls[0][0].content)
        self.assertIn("### Memory: team.md", responder.calls[0][0].content)

    def test_acknowledgement_loop_starts_and_stops_on_search(self) -> None:
        loop = FakeAcknowledgementLoop()
        controller = AssistantController(
            prompt_builder=PromptBuilder(
                assistant_name="Amy",
                project_context="",
                wake_word="amy",
            ),
            responder=FakeResponder(),
            speaker=FakeSpeaker(),
            wake_word="amy",
            web_search=FakeWebSearch(),
            acknowledgment_callback=loop.start,
            acknowledgment_stop_callback=loop.stop,
        )

        controller.process_transcript("amy search for python dataclasses")

        self.assertEqual(loop.started, 1)
        self.assertEqual(loop.stopped, 1)

    def test_acknowledgement_loop_stops_before_reply_speaks(self) -> None:
        loop = FakeAcknowledgementLoop()
        speaker = StopBeforeSpeakSpeaker(loop)
        controller = AssistantController(
            prompt_builder=PromptBuilder(
                assistant_name="Amy",
                project_context="",
                wake_word="amy",
            ),
            responder=FakeResponder(),
            speaker=speaker,
            wake_word="amy",
            web_search=FakeWebSearch(),
            acknowledgment_callback=loop.start,
            acknowledgment_stop_callback=loop.stop,
        )

        controller.process_transcript("amy search for python dataclasses")

        self.assertTrue(speaker.saw_stopped)

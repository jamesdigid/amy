from __future__ import annotations

import threading
import unittest

from agents.amy.memory import OpenAIMemoryClassifier
from agents.amy.models import Message
from agents.providers.openai import OpenAIResponder


class _FakeChoiceMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeChoiceMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = _FakeChat(content)


class OpenAIProviderTests(unittest.TestCase):
    def test_responder_generate_reply_uses_chat_completion_payload(self) -> None:
        responder = OpenAIResponder(api_key="test", model="gpt-4.1-mini")
        fake_client = _FakeClient("hello there")
        responder._client = fake_client  # type: ignore[assignment]

        reply = responder.generate_reply(
            [Message(role="system", content="context"), Message(role="user", content="prompt")],
            threading.Event(),
        )

        self.assertEqual(reply, "hello there")
        self.assertEqual(
            fake_client.chat.completions.calls,
            [
                {
                    "model": "gpt-4.1-mini",
                    "messages": [
                        {"role": "system", "content": "context"},
                        {"role": "user", "content": "prompt"},
                    ],
                    "max_tokens": 300,
                    "temperature": 0.2,
                }
            ],
        )

    def test_memory_classifier_parses_json_decision(self) -> None:
        classifier = OpenAIMemoryClassifier(api_key="test", model="gpt-4.1-mini")
        fake_client = _FakeClient(
            '{"should_save_memory": true, "subject": "vim", "confidence": 0.91, "reason": "stable preference"}'
        )
        classifier._client = fake_client  # type: ignore[assignment]

        decision = classifier.classify("my favorite editor is vim", threading.Event())

        self.assertTrue(decision.should_save)
        self.assertEqual(decision.subject, "vim")
        self.assertEqual(decision.confidence, 0.91)
        self.assertEqual(decision.reason, "stable preference")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from amy.context import PromptBuilder
from amy.models import ConversationTurn


class PromptBuilderTests(unittest.TestCase):
    def test_includes_project_context_and_recent_turns(self) -> None:
        builder = PromptBuilder(
            assistant_name="Amy",
            project_context="Prefer concise answers.",
            wake_word="amy",
            recent_turns=2,
        )
        turns = [
            ConversationTurn(role="user", content="old question"),
            ConversationTurn(role="assistant", content="old answer"),
            ConversationTurn(role="user", content="recent question"),
            ConversationTurn(role="assistant", content="recent answer"),
        ]

        messages = builder.build_messages(turns, "new request")

        self.assertEqual(messages[0].role, "system")
        self.assertIn("Prefer concise answers.", messages[0].content)
        self.assertEqual(
            [message.content for message in messages[-3:]],
            ["recent question", "recent answer", "new request"],
        )

    def test_includes_web_context_when_provided(self) -> None:
        builder = PromptBuilder(
            assistant_name="Amy",
            project_context="Prefer concise answers.",
            wake_word="amy",
        )

        messages = builder.build_messages([], "new request", web_context="Search query: x")

        self.assertIn("Current web context", messages[0].content)
        self.assertIn("Search query: x", messages[0].content)
        self.assertIn("Never read raw URL links aloud", messages[0].content)
        self.assertIn("Treat every interaction as live voice conversation", messages[0].content)

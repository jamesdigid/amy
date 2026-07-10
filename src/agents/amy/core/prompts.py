from __future__ import annotations

from dataclasses import dataclass

from .models import ConversationTurn, Message


@dataclass
class PromptBuilder:
    assistant_name: str
    project_context: str
    wake_word: str
    recent_turns: int = 6

    def build_messages(
        self,
        turns: list[ConversationTurn],
        user_text: str,
        web_context: str = "",
        memory_context: str = "",
    ) -> list[Message]:
        system_prompt = self._build_system_prompt()
        if memory_context:
            system_prompt = f"{system_prompt}\n\nRelevant memories:\n{memory_context}".strip()
        if web_context:
            system_prompt = f"{system_prompt}\n\nCurrent web context:\n{web_context}".strip()
        messages: list[Message] = [Message(role="system", content=system_prompt)]

        recent_history = turns[-self.recent_turns :]
        for turn in recent_history:
            messages.append(Message(role=turn.role, content=turn.content))

        messages.append(Message(role="user", content=user_text))
        return messages

    def _build_system_prompt(self) -> str:
        pieces = [
            f"You are {self.assistant_name}, a concise local voice assistant.",
            f"The wake word is '{self.wake_word}'.",
            "Use the supplied project context to shape tone, priorities, and output format.",
            "When the user asks you to remember something, treat it as durable memory that can be stored across sessions.",
            "Treat every interaction as live voice conversation, not typed chat.",
            "Keep responses short enough to speak naturally, and prefer one clear question at a time.",
            "If you need a follow-up, ask exactly one short question and stop.",
            "Do not continue asking follow-up questions unless the user clearly invites a back-and-forth exchange.",
            "Answer directly with short, useful, cost-conscious responses unless the user asks for depth.",
            "Do not add conversational filler, pleasantries, or follow-up questions unless the user explicitly asks for them.",
            "Do not say things like 'got it', 'thanks', or 'can I assist you today' unless the user requests a follow-up.",
            "Never read raw URL links aloud. If web context includes source details, summarize the content instead of speaking the URL.",
            "When using web context, prefer the fetched article text and cite sources by title rather than reading links aloud.",
        ]
        if self.project_context:
            pieces.append("Project context:")
            pieces.append(self.project_context)
        return "\n\n".join(pieces).strip()

__all__ = ["PromptBuilder"]

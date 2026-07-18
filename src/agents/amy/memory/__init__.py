from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ..core.protocols import MemoryClassifierProtocol, MemoryStoreProtocol

if TYPE_CHECKING:
    import threading

MAX_MEMORY_TAGS = 10
MAX_MEMORY_STEM_LENGTH = 100
MAX_DRAFT_TAGS = 5

_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "be",
    "can",
    "could",
    "don't",
    "dont",
    "for",
    "from",
    "future",
    "hmm",
    "eh",
    "ehh",
    "erm",
    "i",
    "if",
    "in",
    "is",
    "it",
    "just",
    "kind",
    "kinda",
    "keep",
    "later",
    "like",
    "me",
    "my",
    "okay",
    "note",
    "of",
    "on",
    "please",
    "really",
    "remember",
    "save",
    "so",
    "should",
    "that",
    "the",
    "this",
    "to",
    "up",
    "uh",
    "uhh",
    "um",
    "we",
    "will",
    "with",
    "you",
    "your",
    "was",
    "were",
}


@dataclass(frozen=True)
class MemoryEntry:
    path: Path
    tags: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class MemoryDraft:
    path: Path
    tags: tuple[str, ...]
    summary: str
    memories: tuple[str, ...]
    retrieval_notes: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class MemoryDecision:
    should_save: bool
    subject: str = ""
    confidence: float = 0.0
    reason: str = ""


@dataclass
class OpenAIMemoryClassifier:
    api_key: str
    model: str
    max_output_tokens: int = 120
    temperature: float = 0.0
    _client: object | None = field(default=None, init=False, repr=False)

    def classify(self, prompt: str, cancel_event: "threading.Event") -> MemoryDecision:
        if cancel_event.is_set():
            return MemoryDecision(should_save=False)

        client = self._get_client()
        messages = [
            {
                "role": "system",
                "content": (
                    "You decide whether a user utterance should be saved as a durable memory for future sessions. "
                    "Only save facts, preferences, relationships, reminders, or other stable information. "
                    "Do not save filler, transient chat, or general conversation. "
                    "Return only JSON with keys should_save_memory (boolean), subject (string), confidence (number), and reason (string). "
                    "Keep subject concise and free of filler words."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        response = client.chat.completions.create(
            model=self.model,
            messages=cast(list[dict[str, str]], messages),
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        if cancel_event.is_set():
            return MemoryDecision(should_save=False)

        content = response.choices[0].message.content or ""
        return self._parse_decision(content)

    def _get_client(self) -> object:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _parse_decision(self, content: str) -> MemoryDecision:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        payload_text = match.group(0) if match is not None else content.strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return MemoryDecision(should_save=False)

        should_save = bool(payload.get("should_save_memory", False))
        subject = str(payload.get("subject", "")).strip()
        reason = str(payload.get("reason", "")).strip()
        confidence_raw = payload.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        return MemoryDecision(
            should_save=should_save,
            subject=subject,
            confidence=confidence,
            reason=reason,
        )


@dataclass
class MemoryStore:
    memory_dir: Path
    template_name: str = "memory.md"
    max_matches: int = 3

    def retrieve_context(self, prompt: str, limit: int = 3) -> str:
        matches = self.retrieve(prompt, limit=limit)
        if not matches:
            return ""
        sections = [self._format_entry(entry) for entry in matches]
        return "\n\n".join(sections).strip()

    def draft_from_prompt(self, prompt: str, subject: str | None = None) -> MemoryDraft | None:
        subject_text = self._extract_memory_subject(prompt) if subject is None else subject.strip()
        subject_text = self._normalize_subject(subject_text)
        if not subject_text:
            return None

        tags = self._draft_tags(subject_text)
        if not tags:
            return None

        summary = self._build_summary(subject_text)
        memories = (self._build_memory_statement(subject_text),)
        retrieval_notes = (
            "Created from an explicit request to remember this for the future.",
        )
        content = self._render_memory_markdown(summary, memories, retrieval_notes)
        path = self.memory_dir / f"{'.'.join(tags)}.md"
        return MemoryDraft(
            path=path,
            tags=tags,
            summary=summary,
            memories=memories,
            retrieval_notes=retrieval_notes,
            content=content,
        )

    def save_draft(self, draft: MemoryDraft) -> Path:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if draft.path.exists():
            existing = draft.path.read_text(encoding="utf-8").rstrip()
            if existing:
                merged = f"{existing}\n\n---\n\n{draft.content.strip()}\n"
            else:
                merged = f"{draft.content.strip()}\n"
        else:
            merged = f"{draft.content.strip()}\n"
        draft.path.write_text(merged, encoding="utf-8")
        return draft.path

    def retrieve(self, prompt: str, limit: int = 3) -> list[MemoryEntry]:
        prompt_terms = self._tokenize(prompt)
        if not prompt_terms or not self.memory_dir.exists():
            return []

        entries = self._load_entries()
        scored_entries = [
            (self._score_entry(entry, prompt_terms), entry)
            for entry in entries
        ]
        matched_entries = [
            entry
            for score, entry in scored_entries
            if score > 0
        ]
        matched_entries.sort(
            key=lambda entry: (
                -self._score_entry(entry, prompt_terms),
                entry.path.name.lower(),
            )
        )
        return matched_entries[: max(0, min(limit, self.max_matches))]

    def _load_entries(self) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        for path in sorted(self.memory_dir.glob("*.md"), key=lambda item: item.name.lower()):
            if path.name == self.template_name:
                continue
            if not path.is_file():
                continue
            entries.append(
                MemoryEntry(
                    path=path,
                    tags=self._derive_tags(path),
                    content=path.read_text(encoding="utf-8").strip(),
                )
            )
        return entries

    def _derive_tags(self, path: Path) -> tuple[str, ...]:
        stem = path.stem.lower()
        if len(stem) > MAX_MEMORY_STEM_LENGTH:
            return ()

        parts = [part for part in stem.split(".") if part]
        if not parts:
            return (stem,)
        if len(parts) > MAX_MEMORY_TAGS:
            return ()
        return tuple(dict.fromkeys(parts))

    def _draft_tags(self, subject: str) -> tuple[str, ...]:
        tokens = self._ordered_tokens(subject)
        tags: list[str] = []
        for token in tokens:
            if token in _STOP_WORDS:
                continue
            if token in tags:
                continue
            tags.append(token)
            if len(tags) >= MAX_DRAFT_TAGS:
                break
        if not tags:
            return ()

        stem = ".".join(tags)
        if len(stem) > MAX_MEMORY_STEM_LENGTH:
            return ()
        return tuple(tags)

    def _score_entry(self, entry: MemoryEntry, prompt_terms: set[str]) -> int:
        return sum(1 for tag in entry.tags if tag in prompt_terms)

    def _format_entry(self, entry: MemoryEntry) -> str:
        tags_text = ", ".join(entry.tags)
        sections = [
            f"### Memory: {entry.path.name}",
            f"Tags: {tags_text}",
            "",
            entry.content,
        ]
        return "\n".join(sections).strip()

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}

    def _ordered_tokens(self, text: str) -> list[str]:
        return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token]

    def _extract_memory_subject(self, prompt: str) -> str:
        pattern = re.compile(
            r"(?:remember this|remember that|remember for later|save this for later|don't forget|dont forget|note this|keep this in mind)\s*[:,-]?\s*(.*)",
            re.IGNORECASE,
        )
        match = pattern.search(prompt)
        if match is not None:
            return match.group(1).strip(" ,.:;-")
        return prompt.strip()

    def _build_summary(self, subject: str) -> str:
        summary = subject.strip()
        if not summary:
            return "User asked Amy to remember a future note."
        return summary[:1].upper() + summary[1:]

    def _build_memory_statement(self, subject: str) -> str:
        statement = subject.strip()
        if not statement:
            return "Amy should remember this future note."
        return statement[:1].upper() + statement[1:]

    def _render_memory_markdown(
        self,
        summary: str,
        memories: tuple[str, ...],
        retrieval_notes: tuple[str, ...],
    ) -> str:
        memory_lines = "\n".join(f"- {memory}" for memory in memories)
        retrieval_lines = "\n".join(f"- {note}" for note in retrieval_notes)
        return (
            "# Memory\n\n"
            "## Summary\n"
            f"{summary}\n\n"
            "## Memories\n"
            f"{memory_lines}\n\n"
            "## Retrieval Notes\n"
            f"{retrieval_lines}"
        ).strip()

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _normalize_subject(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text.strip())
        cleaned = cleaned.strip(" \t\r\n,.:;\"'")
        return cleaned

__all__ = [
    "MemoryClassifierProtocol",
    "MemoryDecision",
    "MemoryDraft",
    "MemoryEntry",
    "MemoryStore",
    "MemoryStoreProtocol",
    "OpenAIMemoryClassifier",
]

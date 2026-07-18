# Amy AI Voice Assistant

Local Python voice assistant that wakes on `amy`, transcribes speech locally, can pull in lightweight web search for current topics, sends text plus project context to OpenAI, and speaks responses back locally.

## Requirements
- macOS
- Python 3.10 or newer
- An OpenAI API key in `OPENAI_API_KEY`

## Quick Start
The repo now supports a single bootstrap script:

```bash
./scripts/amy setup
```

That creates the local virtual environment and installs the assistant with audio and developer dependencies.

Before you run Amy, create a local `.env` file from the example:

```bash
cp .env.example .env
```

Then add your OpenAI key to `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key
```

To deploy Amy in the background after setup:

```bash
./scripts/amy deploy
```

## Lifecycle Commands
- `./scripts/amy run` starts Amy in the foreground with the interactive command loop.
- `./scripts/amy setup` creates `.venv` and installs dependencies.
- `./scripts/amy start` launches Amy in the background.
- `./scripts/amy stop` stops the background process.
- `./scripts/amy status` reports whether the background process is running.
- `./scripts/amy deploy` runs setup if needed and then starts Amy.

After setup, you can also use the installed console script inside the venv:

```bash
amy run
```

## Test Suite
Run the full test suite with:

```bash
uv run pytest
```

If you have already run `./scripts/amy setup`, you can also run tests from the local virtual environment:

```bash
./.venv/bin/pytest
```

## Configuration
Optional environment variables:
- `AMY_MODEL`
- `AMY_ASSISTANT_NAME`
- `AMY_CONTEXT_PATH`
- `AMY_MEMORY_DIR`
- `AMY_RECENT_TURNS`
- `AMY_WAKE_WORD`
- `AMY_TRANSCRIPT_LANGUAGE`
- `AMY_LOG_TRANSCRIPTS`

You can also edit `config/project_context.md` to shape Amy's tone and behavior for your project.

## What To Expect
- Amy runs locally and uses terminal commands for lifecycle control.
- Say `amy` to begin a voice interaction.
- Say `amy status check` or `check your status` to ask Amy for her current runtime status, registered skills, a lightweight smoke test, and relevant skill notes.
- After Amy responds, she stays in listening mode for about 10 seconds so you can follow up without repeating the wake word.
- Use the terminal commands `pause`, `resume`, `status`, and `quit` while running in the foreground.
- Ask current or lookup-style questions and Amy will add basic web search context automatically.
- Amy can also retrieve matching markdown memories from `src/agents/amy/memory` when your prompt terms match the dot-delimited file tags.
- Say things like `remember that...`, `remember this...`, or `don't forget...` to make Amy consider saving a future memory.
- Local speech-to-text and local text-to-speech keep OpenAI usage text-only and cost-effective.
- Set `AMY_LOG_TRANSCRIPTS=1` if you want Amy to log the raw transcripts she hears.

## Notes
- Background start/stop state is stored under `.amy/`.
- `.env` is loaded automatically by `./scripts/amy` when it exists.
- If `python3` is not available, install Python 3.10+ first and rerun `./scripts/amy setup`.
- For microphone access during a call, use `pause` so Amy releases the channel immediately.

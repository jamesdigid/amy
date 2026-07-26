# Description

Amy is an open-source AI assistant exploring how agents can learn, accumulate, and compose new capabilities over time.

Rather than relying solely on predefined tools and workflows, Amy aims to capture successful outcomes and transform them into reusable capabilities that compound with experience.

The current implementation focuses on a fast, local-first voice experience while laying the architectural foundation for adaptive, self-improving agents.

---

## Why Amy?

After exploring countless AI agent frameworks, I noticed a common pattern: most agents are built around predefined tools and workflows. While these systems are powerful, expanding their capabilities often requires humans to continually add new integrations, prompts, or code.

This creates a fundamental limitation. Agents can reason through novel problems, but very little of what they discover becomes structured knowledge that can be reused later. Every interaction starts from nearly the same place.

Amy is built around a different idea.

Instead of viewing an agent as a collection of predefined tools, Amy treats every successful outcome as an opportunity to build a reusable capability. The goal is to create agents that don't simply execute workflows—they continuously expand what they're capable of accomplishing.

---

## Vision

The long-term vision for Amy is to build agents whose capabilities compound through experience.

Today's frontier models can often solve problems they've never seen before through reasoning alone. However, those solutions are typically ephemeral. They solve a task, consume tokens, and move on. The reasoning that led to success rarely becomes structured knowledge that can be reused, composed, or improved over time.

Amy explores the infrastructure required to change that.

Rather than treating every request as a brand new reasoning problem, Amy aims to capture successful outcomes and evolve them into reusable capabilities. Those capabilities should be discoverable, composable, adaptable, and continuously refined as the agent gains more experience.

A capability might begin as something simple—checking email for new sales leads—but the same principles should scale to discovering new business workflows, integrating with unfamiliar software, automating research, or composing multiple capabilities together to solve increasingly complex tasks.

As the implementation evolves, the underlying execution may change. A capability might move from pure reasoning, to a reusable workflow, to a dedicated integration, or eventually to a fully autonomous system. What remains constant is the desired outcome—not the implementation itself.

Amy exists to explore the infrastructure needed for agents that continuously learn from experience, accumulate reusable capabilities, and become more capable over time.

## Requirements
- macOS
- brew
- Python 3.10 or newer
- An OpenAI API key in `OPENAI_API_KEY`

## Quick Start
Before you run Amy, create a local `.env` file from the example:

```bash
cp .env.example .env
```

Then add your OpenAI key to `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key
```

Then run the bootstrap script:

```bash
./scripts/amy setup
```

That creates the local virtual environment, installs the assistant with audio and developer dependencies, and prefetches the default transcription model so the first live run is faster.
If `ffmpeg` is missing on macOS, the setup step will install it through Homebrew automatically.

To start Amy in the background after setup:

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

`./scripts/amy run` is the foreground, interactive command loop. `./scripts/amy deploy` is the background launcher, and it will run setup first if needed before starting Amy as a background process.

After setup, you can also use the installed console script inside the venv:

```bash
./scripts/amy run
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
- `AMY_TRANSCRIPTION_MODEL`
- `AMY_LOG_TRANSCRIPTS`
- `AMY_AUDIO_INPUT_DEVICE`

You can also edit `config/project_context.md` to shape Amy's tone and behavior for your project.

## What To Expect
- Amy runs locally and uses terminal commands for lifecycle control.
- Say `amy` to begin a voice interaction.
- Say `amy status check` or `check your status` to ask Amy for her current runtime status, registered skills, a lightweight smoke test, and relevant skill notes.
- After Amy responds, she stays in listening mode for about 10 seconds so you can follow up without repeating the wake word.
- Voice has no control words. Use the terminal commands `pause`, `resume`, `status`, and `quit` while running in the foreground.
- Background Amy has no pause/resume channel; use `stop` and restart it if you need to release the microphone.
- Ask current or lookup-style questions and Amy will add basic web search context automatically.
- Amy can also retrieve matching markdown memories from `src/agents/amy/memory` when your prompt terms match the dot-delimited file tags.
- Say things like `remember that...`, `remember this...`, or `don't forget...` to make Amy consider saving a future memory.
- Local speech-to-text uses MLX Whisper on Apple Silicon and caches the transcription model under `~/.cache/huggingface`.
- The default transcription model is `mlx-community/whisper-large-v3-turbo`, which is downloaded during setup if possible.
- Local text-to-speech keeps OpenAI usage text-only and cost-effective.
- Set `AMY_LOG_TRANSCRIPTS=1` if you want Amy to log the raw transcripts she hears.
- Set `AMY_AUDIO_INPUT_DEVICE` to the exact device name or PortAudio index if you want Amy to capture from a specific audio interface instead of the system default input.

## Notes
- Background start/stop state is stored under `.amy/`.
- `.env` is loaded automatically by `./scripts/amy` when it exists.
- If the model prefetch fails during setup, Amy will still work and download the transcription model on first run.
- If `python3` is not available, install Python 3.10+ first and rerun `./scripts/amy setup`.
- For microphone access during a call in the foreground, use `pause` so Amy releases the channel immediately. In the background, stop Amy and start it again when you are ready.

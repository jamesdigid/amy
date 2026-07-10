# Amy AI Voice Assistant

Local Python voice assistant that wakes on `amy`, transcribes speech locally, can pull in lightweight web search for current topics, sends text plus project context to OpenAI, and speaks responses back locally.

## Requirements
- macOS
- Python 3.10 or newer (3.11 recommended)
- An OpenAI API key in `OPENAI_API_KEY`

## Bring It Online
1. Create and activate a Python 3.11 virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
2. Upgrade packaging tools and install the assistant:
   ```bash
   python -m pip install --upgrade pip
   pip install -e ".[audio,dev]"
   ```
3. Set your OpenAI key:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```
4. Optionally edit `config/project_context.md` to shape Amy's tone and behavior for your project.
5. Store durable memories in `memory/*.md` and use dot-delimited filename tags for retrieval.
   - `memory/memory.md` is the editable template for the memory format.
   - Keep filenames to at most 10 tags and under 100 characters in the stem.
   - Amy uses an LLM classifier to decide when something should become a durable memory, then writes it through the app’s file I/O layer.
6. Start the assistant:
   ```bash
   python -m amy
   ```

## What To Expect
- The system runs as a terminal-controlled local assistant, not a background service.
- Say `amy` to begin a voice interaction.
- After Amy responds, she stays in listening mode for about 10 seconds so you can follow up without repeating the wake word.
- Use the terminal commands `pause`, `resume`, `status`, and `quit` to control the channel.
- Ask current or lookup-style questions and Amy will add basic web search context automatically.
- Amy can also retrieve matching markdown memories from `memory` when your prompt terms match the dot-delimited file tags.
- Say things like `remember that...`, `remember this...`, or `don't forget...` to make Amy consider saving a future memory.
- Local speech-to-text and local text-to-speech keep OpenAI usage text-only and cost-effective.
- Set `AMY_LOG_TRANSCRIPTS=1` if you want Amy to log the raw transcripts she hears.

## Notes
- If `python3.11` is not available, install Python 3.11 first and rerun the steps above.
- The default `python3` on this machine is 3.9, which is too old for this project.
- For microphone access during a call, use `pause` so Amy releases the channel immediately.

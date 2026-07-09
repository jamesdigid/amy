# Amy Research Assistant

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
5. Start the assistant:
   ```bash
   python -m amy
   ```

## What To Expect
- The system runs as a terminal-controlled local assistant, not a background service.
- Say `amy` to begin a voice interaction.
- Use the terminal commands `pause`, `resume`, `status`, and `quit` to control the channel.
- Ask current or lookup-style questions and Amy will add basic web search context automatically.
- Local speech-to-text and local text-to-speech keep OpenAI usage text-only and cost-effective.

## Notes
- If `python3.11` is not available, install Python 3.11 first and rerun the steps above.
- The default `python3` on this machine is 3.9, which is too old for this project.
- For microphone access during a call, use `pause` so Amy releases the channel immediately.

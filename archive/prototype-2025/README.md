# IntuitionOS 🧠

A terminal-style operating system where all commands, scheduling, and reasoning are guided by an intuition model (neural decision layer) instead of deterministic logic.

## Overview

IntuitionOS is an experimental operating system that operates on human-like intuition-based reasoning rather than pure logic. It uses a Large Language Model (LLM) as its core brain, which "thinks aloud" through internal inference steps before acting on user requests.

## Core Features

- **Natural Language Interface**: Communicate with the OS using plain English (or any natural language)
- **Thinking Aloud**: The system shows its reasoning process before executing tasks
- **Persistent Memory**: All conversations, tasks, and facts are stored in JSON format
- **Intuition-Based Decision Making**: Actions are based on contextual understanding, not just deterministic rules
- **Interactive Shell**: Text-based terminal interface for seamless interaction

## Architecture

### Components

1. **Shell** (`shell.py`): Text-based terminal interface for user interaction
2. **Reasoning Engine** (`reasoning.py`): LLM-powered brain that processes requests and thinks through decisions
3. **Memory** (`memory.py`): JSON-based persistent storage for context, conversations, tasks, and facts
4. **Task Executor** (`executor.py`): Safely executes commands based on reasoning decisions

### Flow

```
User Input → Reasoning Engine → Internal Thinking → Action Decision → Task Executor → Response
                ↓
            Memory Storage (Persistent Context)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/soheilAppear/IntuitionOS.git
cd IntuitionOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

### Starting IntuitionOS

Run the main entry point:
```bash
python main.py
```

### Example Interactions

```
🧠 intuition> What's the current date?

💭 Thinking...
──────────────────────────────────────────────────────────
  Internal Reasoning:
──────────────────────────────────────────────────────────
The user wants to know the current date. I should execute
the system date command to provide this information.

──────────────────────────────────────────────────────────
  Response:
──────────────────────────────────────────────────────────
run: date
[Output shows current date and time]
```

### Special Commands

- `help` - Display help information
- `memory` - Show recent conversations
- `tasks` - Display all stored tasks
- `facts` - Show all learned facts
- `clear` - Clear all memory
- `exit` or `quit` - Exit IntuitionOS

### Natural Language Examples

You can interact with IntuitionOS naturally:

- "Create a task to learn Python"
- "What tasks do I have?"
- "Tell me an interesting fact about AI"
- "What's my username?"
- "List the files in the current directory"

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini  # or gpt-4, gpt-3.5-turbo, etc.
```

### Memory Storage

All persistent data is stored in `intuition_memory.json` with the following structure:

```json
{
  "conversations": [],
  "tasks": [],
  "facts": [],
  "created_at": "timestamp"
}
```

## Development

### Project Structure

```
IntuitionOS/
├── intuition_os/
│   ├── __init__.py       # Package initialization
│   ├── shell.py          # Terminal interface
│   ├── reasoning.py      # LLM reasoning engine
│   ├── memory.py         # Persistent storage
│   └── executor.py       # Task execution
├── main.py               # Entry point
├── requirements.txt      # Dependencies
├── .env.example          # Example configuration
└── README.md            # Documentation
```

### Adding New Features

1. **New Commands**: Add to `Shell._handle_special_command()` in `shell.py`
2. **Safe System Commands**: Update `TaskExecutor.SAFE_COMMANDS` in `executor.py`
3. **Memory Types**: Extend the `Memory` class in `memory.py`

## Safety

IntuitionOS includes safety features:

- Only whitelisted system commands can be executed
- Commands run with timeout protection
- All actions are logged in memory
- API key stored in environment variables (not in code)

## Future Enhancements

- [ ] Voice input support
- [ ] GUI interface
- [ ] SQLite database backend
- [ ] Multi-user support
- [ ] Plugin system
- [ ] Scheduled tasks
- [ ] Integration with external tools

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This project is inspired by the concept of human-like intuition in decision-making, leveraging modern AI capabilities to create a more natural computing experience.

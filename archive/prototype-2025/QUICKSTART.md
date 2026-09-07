# Quick Start Guide

## IntuitionOS - Get Started in 3 Minutes

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- OpenAI API key (optional, but recommended for full functionality)

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/soheilAppear/IntuitionOS.git
cd IntuitionOS
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure API key (Optional but Recommended):**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

4. **Run the system:**
```bash
python main.py
```

### First Steps

Once IntuitionOS starts, you'll see:
```
🧠 IntuitionOS - Intuition-Based Operating System
```

Try these commands:

1. **Get help:**
   ```
   🧠 intuition> help
   ```

2. **Ask a question:**
   ```
   🧠 intuition> What can you do?
   ```

3. **View memory:**
   ```
   🧠 intuition> memory
   ```

4. **Exit:**
   ```
   🧠 intuition> exit
   ```

### Without API Key

IntuitionOS will run in fallback mode if no API key is configured:
- ✓ All features work
- ✓ Memory is persistent
- ⚠ Reasoning is basic (no LLM intelligence)

### With API Key

With an OpenAI API key configured:
- ✓ Full LLM reasoning
- ✓ Natural language understanding
- ✓ Context-aware responses
- ✓ Intelligent task planning

### Run the Demo

To see all features without interaction:
```bash
python demo.py
```

### Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Experiment with natural language commands
- Check out the code in `intuition_os/` directory
- Contribute to the project!

### Troubleshooting

**Module not found error:**
```bash
pip install -r requirements.txt
```

**API key not working:**
- Check that `.env` file exists
- Verify `OPENAI_API_KEY=sk-...` is set correctly
- Make sure the key is valid

**Memory not persisting:**
- Check file permissions in the directory
- Look for `intuition_memory.json` file

### Getting Help

- Check issues on GitHub
- Read the documentation in README.md
- Run `help` command within IntuitionOS

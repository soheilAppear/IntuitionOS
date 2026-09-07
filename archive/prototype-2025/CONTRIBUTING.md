# Contributing to IntuitionOS

Thank you for your interest in contributing to IntuitionOS! This document provides guidelines and information for contributors.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)

### Suggesting Enhancements

We welcome enhancement suggestions! Please:
- Check if the enhancement has already been suggested
- Provide a clear use case
- Explain how it aligns with the project's goals

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions/classes
   - Update documentation if needed

4. **Test your changes**
   ```bash
   python demo.py
   python main.py
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add: brief description of your changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/soheilAppear/IntuitionOS.git
cd IntuitionOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key (for testing):
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Code Style

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and single-purpose

### Docstring Format

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    pass
```

## Project Structure

```
IntuitionOS/
├── intuition_os/          # Core package
│   ├── memory.py          # Memory management
│   ├── reasoning.py       # LLM reasoning engine
│   ├── executor.py        # Task execution
│   └── shell.py           # Shell interface
├── examples/              # Usage examples
├── main.py               # Entry point
├── demo.py               # Demo script
└── tests/                # Tests (future)
```

## Areas for Contribution

### High Priority
- [ ] Add comprehensive unit tests
- [ ] SQLite backend for memory
- [ ] Voice input support
- [ ] Enhanced error handling

### Medium Priority
- [ ] GUI interface
- [ ] Plugin system
- [ ] Scheduled tasks
- [ ] Multi-user support

### Documentation
- [ ] Video tutorials
- [ ] More examples
- [ ] API reference
- [ ] Architecture diagrams

## Testing

Currently, manual testing is done via:
- `python demo.py` - Feature demonstration
- `python main.py` - Interactive testing
- `python examples/*.py` - Example scripts

**We need help adding automated tests!**

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion
- Reach out to maintainers

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain a positive environment

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to IntuitionOS! 🧠✨

# IntuitionOS Examples

This directory contains examples of how to use and extend IntuitionOS.

## Examples

### 1. Custom Tasks (`custom_tasks.py`)
Demonstrates how to:
- Add custom task types
- Group tasks by status
- Associate facts with tasks
- Use reasoning to prioritize work

**Run:**
```bash
python examples/custom_tasks.py
```

### 2. Library Usage (`library_usage.py`)
Shows how to:
- Use IntuitionOS as a library in your own application
- Process requests programmatically
- Access conversation history
- Integrate with existing code

**Run:**
```bash
python examples/library_usage.py
```

## Creating Your Own Examples

To create a new example:

1. Import the necessary components:
```python
from intuition_os.memory import Memory
from intuition_os.reasoning import ReasoningEngine
from intuition_os.executor import TaskExecutor
```

2. Initialize the components:
```python
memory = Memory(memory_file="example_memory.json")
reasoning = ReasoningEngine()
executor = TaskExecutor()
```

3. Use them in your application logic!

## Contributing

Have a cool example? Submit a PR to add it to this directory!

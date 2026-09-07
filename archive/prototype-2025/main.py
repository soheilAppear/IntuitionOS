#!/usr/bin/env python3
"""
IntuitionOS - Main entry point
A terminal-style OS guided by intuition-based reasoning.
"""
import os
import sys
from dotenv import load_dotenv

from intuition_os.memory import Memory
from intuition_os.reasoning import ReasoningEngine
from intuition_os.executor import TaskExecutor
from intuition_os.shell import Shell


def main():
    """Main entry point for IntuitionOS."""
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    memory = Memory(memory_file="intuition_memory.json")
    
    # Get model from environment or use default
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    reasoning = ReasoningEngine(model=model)
    
    executor = TaskExecutor()
    
    # Start the shell
    shell = Shell(memory, reasoning, executor)
    shell.start()


if __name__ == "__main__":
    main()

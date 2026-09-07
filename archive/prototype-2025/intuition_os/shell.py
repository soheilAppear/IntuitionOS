"""
Shell interface for IntuitionOS - text-based terminal interface.
"""
import sys
from typing import Optional
from .memory import Memory
from .reasoning import ReasoningEngine
from .executor import TaskExecutor


class Shell:
    """Text-based shell interface for IntuitionOS."""
    
    def __init__(self, memory: Memory, reasoning: ReasoningEngine, executor: TaskExecutor):
        """Initialize the shell with core components."""
        self.memory = memory
        self.reasoning = reasoning
        self.executor = executor
        self.running = True
    
    def start(self):
        """Start the interactive shell."""
        self._print_welcome()
        
        while self.running:
            try:
                user_input = input("\n🧠 intuition> ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if self._handle_special_command(user_input):
                    continue
                
                # Process through reasoning engine
                self._process_input(user_input)
                
            except KeyboardInterrupt:
                print("\n\nUse 'exit' to quit IntuitionOS.")
            except EOFError:
                break
        
        self._print_goodbye()
    
    def _print_welcome(self):
        """Print welcome message."""
        print("=" * 60)
        print("  🧠 IntuitionOS - Intuition-Based Operating System")
        print("=" * 60)
        print("\nWelcome! I operate on intuition-based reasoning.")
        print("Type your requests in natural language, and I'll think")
        print("through them before responding.\n")
        print("Commands: exit, help, clear, memory, tasks, facts")
        print("=" * 60)
    
    def _print_goodbye(self):
        """Print goodbye message."""
        print("\n\n👋 Thank you for using IntuitionOS. Goodbye!")
    
    def _handle_special_command(self, command: str) -> bool:
        """Handle special shell commands. Returns True if handled."""
        cmd_lower = command.lower()
        
        if cmd_lower == "exit" or cmd_lower == "quit":
            self.running = False
            return True
        
        elif cmd_lower == "help":
            self._show_help()
            return True
        
        elif cmd_lower == "clear":
            print("\nClearing memory...")
            self.memory.clear()
            print("Memory cleared!")
            return True
        
        elif cmd_lower == "memory":
            self._show_memory()
            return True
        
        elif cmd_lower == "tasks":
            self._show_tasks()
            return True
        
        elif cmd_lower == "facts":
            self._show_facts()
            return True
        
        return False
    
    def _show_help(self):
        """Show help information."""
        print("\n" + "=" * 60)
        print("  IntuitionOS Help")
        print("=" * 60)
        print("\nSpecial Commands:")
        print("  exit, quit  - Exit IntuitionOS")
        print("  help        - Show this help message")
        print("  clear       - Clear all memory")
        print("  memory      - Show recent conversations")
        print("  tasks       - Show all tasks")
        print("  facts       - Show known facts")
        print("\nNatural Language:")
        print("  Just type what you want in plain English!")
        print("  Examples:")
        print("    - What's the current date?")
        print("    - Create a task to learn Python")
        print("    - Tell me a fun fact")
        print("=" * 60)
    
    def _show_memory(self):
        """Show recent conversations."""
        conversations = self.memory.get_recent_conversations()
        if not conversations:
            print("\nNo conversations in memory yet.")
            return
        
        print("\n" + "=" * 60)
        print("  Recent Conversations")
        print("=" * 60)
        for conv in conversations:
            print(f"\n[{conv['timestamp']}]")
            print(f"You: {conv['user_input']}")
            print(f"Response: {conv['response'][:100]}...")
    
    def _show_tasks(self):
        """Show all tasks."""
        tasks = self.memory.get_all_tasks()
        if not tasks:
            print("\nNo tasks in memory.")
            return
        
        print("\n" + "=" * 60)
        print("  All Tasks")
        print("=" * 60)
        for task in tasks:
            print(f"\n[{task['status']}] {task['task']}")
            print(f"  Added: {task['timestamp']}")
    
    def _show_facts(self):
        """Show known facts."""
        facts = self.memory.get_all_facts()
        if not facts:
            print("\nNo facts in memory.")
            return
        
        print("\n" + "=" * 60)
        print("  Known Facts")
        print("=" * 60)
        for fact in facts:
            print(f"\n• {fact['fact']}")
            print(f"  Learned: {fact['timestamp']}")
    
    def _process_input(self, user_input: str):
        """Process user input through the reasoning engine."""
        # Get context from memory
        context = self.memory.context
        
        # Think through the input
        print("\n💭 Thinking...")
        result = self.reasoning.think_and_act(user_input, context)
        
        # Display reasoning
        print("\n" + "─" * 60)
        print("  Internal Reasoning:")
        print("─" * 60)
        print(result["reasoning"])
        
        # Execute action
        response = self.executor.execute(result["action"])
        
        # Display response
        print("\n" + "─" * 60)
        print("  Response:")
        print("─" * 60)
        print(response)
        
        # Save to memory
        self.memory.add_conversation(user_input, result["reasoning"], response)

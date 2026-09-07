"""
Example: Using IntuitionOS as a library

This example shows how to use IntuitionOS components in your own application.
"""
import sys
from pathlib import Path

# Add parent directory to path to import intuition_os
sys.path.insert(0, str(Path(__file__).parent.parent))

from intuition_os.memory import Memory
from intuition_os.reasoning import ReasoningEngine
from intuition_os.executor import TaskExecutor


class MyApplication:
    """Example application using IntuitionOS components."""
    
    def __init__(self):
        """Initialize the application."""
        self.memory = Memory(memory_file="my_app_memory.json")
        self.reasoning = ReasoningEngine()
        self.executor = TaskExecutor()
    
    def process_request(self, user_input: str) -> dict:
        """
        Process a user request and return the result.
        
        Args:
            user_input: Natural language input from user
            
        Returns:
            dict with 'reasoning', 'action', and 'response' keys
        """
        # Get reasoning from the engine
        result = self.reasoning.think_and_act(user_input, self.memory.context)
        
        # Execute the action
        response = self.executor.execute(result['action'])
        
        # Save to memory
        self.memory.add_conversation(user_input, result['reasoning'], response)
        
        return {
            'reasoning': result['reasoning'],
            'action': result['action'],
            'response': response
        }
    
    def get_conversation_history(self, limit: int = 10):
        """Get recent conversation history."""
        return self.memory.get_recent_conversations(limit)


def main():
    """Demonstrate using IntuitionOS as a library."""
    print("=" * 60)
    print("  Using IntuitionOS as a Library")
    print("=" * 60)
    
    # Create application instance
    app = MyApplication()
    
    # Process some requests
    requests = [
        "Hello, how are you?",
        "What's 2 + 2?",
        "Tell me about AI",
    ]
    
    for request in requests:
        print(f"\n📝 User: {request}")
        print("-" * 60)
        
        result = app.process_request(request)
        
        print(f"💭 Reasoning: {result['reasoning'][:80]}...")
        print(f"🎯 Action: {result['action'][:80]}...")
        print(f"💬 Response: {result['response'][:80]}...")
    
    # Show conversation history
    print("\n" + "=" * 60)
    print("  Conversation History")
    print("=" * 60)
    
    history = app.get_conversation_history()
    for i, conv in enumerate(history, 1):
        print(f"\n{i}. {conv['user_input']}")
        print(f"   Response: {conv['response'][:60]}...")
    
    print("\n✅ Library example complete!")
    
    # Clean up
    import os
    if os.path.exists("my_app_memory.json"):
        os.remove("my_app_memory.json")


if __name__ == "__main__":
    main()

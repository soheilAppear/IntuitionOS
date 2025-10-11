#!/usr/bin/env python3
"""
Demo script for IntuitionOS - demonstrates key features without requiring API key.
"""
from intuition_os.memory import Memory
from intuition_os.reasoning import ReasoningEngine
from intuition_os.executor import TaskExecutor


def demo():
    """Run a demonstration of IntuitionOS features."""
    print("=" * 70)
    print("  IntuitionOS Feature Demonstration")
    print("=" * 70)
    
    # Initialize components
    memory = Memory(memory_file="demo_memory.json")
    reasoning = ReasoningEngine()
    executor = TaskExecutor()
    
    print("\n1. Memory System")
    print("-" * 70)
    print("   Creating sample conversations, tasks, and facts...")
    
    # Add sample data
    memory.add_conversation(
        "What is IntuitionOS?",
        "The user is asking about the system itself. I should explain its purpose.",
        "IntuitionOS is an intuition-based operating system that uses AI for reasoning."
    )
    memory.add_task("Learn Python programming", "in_progress")
    memory.add_task("Build a simple AI project", "pending")
    memory.add_fact("IntuitionOS uses LLMs for decision making")
    
    print("   ✓ Data stored successfully")
    
    # Show memory
    print("\n2. Recent Conversations")
    print("-" * 70)
    for conv in memory.get_recent_conversations():
        print(f"   User: {conv['user_input']}")
        print(f"   Response: {conv['response'][:60]}...")
    
    print("\n3. Tasks")
    print("-" * 70)
    for task in memory.get_all_tasks():
        print(f"   [{task['status']}] {task['task']}")
    
    print("\n4. Facts")
    print("-" * 70)
    for fact in memory.get_all_facts():
        print(f"   • {fact['fact']}")
    
    print("\n5. Reasoning Engine (Fallback Mode)")
    print("-" * 70)
    result = reasoning.think_and_act("Hello!", memory.context)
    print(f"   Input: Hello!")
    print(f"   Reasoning: {result['reasoning'][:60]}...")
    print(f"   Action: {result['action'][:60]}...")
    
    print("\n6. Task Executor")
    print("-" * 70)
    response = executor.execute("This is a simple response")
    print(f"   Response: {response[:60]}...")
    
    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70)
    print("\nTo run IntuitionOS interactively, use: python main.py")
    print("For full AI reasoning, set OPENAI_API_KEY in .env file")
    print("=" * 70)
    
    # Clean up demo memory
    import os
    if os.path.exists("demo_memory.json"):
        os.remove("demo_memory.json")


if __name__ == "__main__":
    demo()

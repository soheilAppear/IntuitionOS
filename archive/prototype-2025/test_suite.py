#!/usr/bin/env python3
"""
Comprehensive test script for IntuitionOS
Tests all major components and features.
"""
import sys
import os
from pathlib import Path

# Ensure we can import from parent directory
sys.path.insert(0, str(Path(__file__).parent))

from intuition_os.memory import Memory
from intuition_os.reasoning import ReasoningEngine
from intuition_os.executor import TaskExecutor


def test_memory():
    """Test memory functionality."""
    print("\n" + "=" * 70)
    print("Testing Memory Module")
    print("=" * 70)
    
    memory = Memory(memory_file="/tmp/test_memory.json")
    
    # Test conversation storage
    memory.add_conversation("Test input", "Test reasoning", "Test response")
    convos = memory.get_recent_conversations()
    assert len(convos) == 1, "Failed to add conversation"
    assert convos[0]["user_input"] == "Test input", "Wrong conversation data"
    print("✓ Conversation storage works")
    
    # Test task storage
    memory.add_task("Test task", "pending")
    tasks = memory.get_all_tasks()
    assert len(tasks) == 1, "Failed to add task"
    assert tasks[0]["task"] == "Test task", "Wrong task data"
    print("✓ Task storage works")
    
    # Test fact storage
    memory.add_fact("Test fact")
    facts = memory.get_all_facts()
    assert len(facts) == 1, "Failed to add fact"
    assert facts[0]["fact"] == "Test fact", "Wrong fact data"
    print("✓ Fact storage works")
    
    # Test clear
    memory.clear()
    assert len(memory.get_recent_conversations()) == 0, "Failed to clear memory"
    print("✓ Memory clear works")
    
    # Clean up
    if os.path.exists("/tmp/test_memory.json"):
        os.remove("/tmp/test_memory.json")
    
    print("✅ Memory module: ALL TESTS PASSED")


def test_reasoning():
    """Test reasoning engine."""
    print("\n" + "=" * 70)
    print("Testing Reasoning Engine")
    print("=" * 70)
    
    reasoning = ReasoningEngine()
    
    # Test with fallback mode (no API key)
    result = reasoning.think_and_act("Hello", {})
    assert "reasoning" in result, "Missing reasoning in result"
    assert "action" in result, "Missing action in result"
    print("✓ Reasoning engine returns expected structure")
    
    # Test with context
    context = {
        "conversations": [{"user_input": "Previous", "response": "Response"}],
        "tasks": [{"task": "Do something", "status": "pending"}],
        "facts": [{"fact": "A fact"}]
    }
    result = reasoning.think_and_act("What should I do?", context)
    assert result["reasoning"] is not None, "No reasoning provided"
    assert result["action"] is not None, "No action provided"
    print("✓ Reasoning engine handles context")
    
    print("✅ Reasoning engine: ALL TESTS PASSED")


def test_executor():
    """Test task executor."""
    print("\n" + "=" * 70)
    print("Testing Task Executor")
    print("=" * 70)
    
    executor = TaskExecutor()
    
    # Test simple response
    result = executor.execute("Simple response text")
    assert result == "Simple response text", "Failed to handle simple response"
    print("✓ Executor handles simple responses")
    
    # Test safe command execution would go here
    # (skipped for now as it requires specific environment)
    
    print("✅ Task executor: ALL TESTS PASSED")


def test_integration():
    """Test integration of all components."""
    print("\n" + "=" * 70)
    print("Testing Integration")
    print("=" * 70)
    
    # Initialize all components
    memory = Memory(memory_file="/tmp/test_integration_memory.json")
    reasoning = ReasoningEngine()
    executor = TaskExecutor()
    
    # Simulate a complete flow
    user_input = "Test integration"
    
    # Get reasoning
    result = reasoning.think_and_act(user_input, memory.context)
    
    # Execute action
    response = executor.execute(result["action"])
    
    # Save to memory
    memory.add_conversation(user_input, result["reasoning"], response)
    
    # Verify everything worked
    convos = memory.get_recent_conversations()
    assert len(convos) == 1, "Integration: conversation not saved"
    assert convos[0]["user_input"] == user_input, "Integration: wrong input saved"
    
    print("✓ Complete workflow integration works")
    
    # Clean up
    if os.path.exists("/tmp/test_integration_memory.json"):
        os.remove("/tmp/test_integration_memory.json")
    
    print("✅ Integration: ALL TESTS PASSED")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  IntuitionOS - Comprehensive Test Suite")
    print("=" * 70)
    
    try:
        test_memory()
        test_reasoning()
        test_executor()
        test_integration()
        
        print("\n" + "=" * 70)
        print("  🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70)
        print("\nIntuitionOS is working correctly!")
        print("Run 'python main.py' to start the interactive shell.")
        print("=" * 70 + "\n")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

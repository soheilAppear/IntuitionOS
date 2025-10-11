"""
Example: Extending IntuitionOS with custom task handlers

This example shows how to add custom task types to IntuitionOS.
"""
import sys
from pathlib import Path

# Add parent directory to path to import intuition_os
sys.path.insert(0, str(Path(__file__).parent.parent))

from intuition_os.memory import Memory
from intuition_os.reasoning import ReasoningEngine


def custom_task_handler():
    """Example of handling custom tasks with IntuitionOS."""
    
    # Initialize memory
    memory = Memory(memory_file="custom_example_memory.json")
    
    # Add some custom tasks
    print("Adding custom tasks...")
    memory.add_task("Build a web scraper", "planned")
    memory.add_task("Write documentation", "in_progress")
    memory.add_task("Deploy to production", "pending")
    
    # Add related facts
    memory.add_fact("Web scraping requires BeautifulSoup library")
    memory.add_fact("Documentation should include examples")
    
    # Show tasks by status
    print("\n📋 Tasks by Status:")
    print("-" * 50)
    
    tasks = memory.get_all_tasks()
    status_groups = {}
    
    for task in tasks:
        status = task['status']
        if status not in status_groups:
            status_groups[status] = []
        status_groups[status].append(task)
    
    for status, task_list in status_groups.items():
        print(f"\n{status.upper()}:")
        for task in task_list:
            print(f"  • {task['task']}")
    
    # Show related facts
    print("\n💡 Related Facts:")
    print("-" * 50)
    for fact in memory.get_all_facts():
        print(f"  • {fact['fact']}")
    
    # Simulate reasoning about next task
    print("\n🧠 Reasoning about next action...")
    print("-" * 50)
    
    reasoning = ReasoningEngine()
    result = reasoning.think_and_act(
        "What task should I work on next?",
        memory.context
    )
    
    print(f"Reasoning: {result['reasoning']}")
    print(f"Action: {result['action']}")
    
    # Clean up
    import os
    if os.path.exists("custom_example_memory.json"):
        os.remove("custom_example_memory.json")
    
    print("\n✅ Example complete!")


if __name__ == "__main__":
    custom_task_handler()

"""
Memory module for IntuitionOS - handles persistent context storage.
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Any


class Memory:
    """Persistent memory storage for IntuitionOS using JSON."""
    
    def __init__(self, memory_file: str = "memory.json"):
        """Initialize memory with a JSON file."""
        self.memory_file = memory_file
        self.context = self._load_memory()
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from JSON file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._create_default_memory()
        return self._create_default_memory()
    
    def _create_default_memory(self) -> Dict[str, Any]:
        """Create default memory structure."""
        return {
            "conversations": [],
            "tasks": [],
            "facts": [],
            "created_at": datetime.now().isoformat()
        }
    
    def save(self):
        """Save current memory state to JSON file."""
        with open(self.memory_file, 'w') as f:
            json.dump(self.context, f, indent=2)
    
    def add_conversation(self, user_input: str, reasoning: str, response: str):
        """Add a conversation to memory."""
        self.context["conversations"].append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "reasoning": reasoning,
            "response": response
        })
        self.save()
    
    def add_task(self, task: str, status: str = "pending"):
        """Add a task to memory."""
        self.context["tasks"].append({
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "status": status
        })
        self.save()
    
    def add_fact(self, fact: str):
        """Add a fact to memory."""
        self.context["facts"].append({
            "timestamp": datetime.now().isoformat(),
            "fact": fact
        })
        self.save()
    
    def get_recent_conversations(self, limit: int = 5) -> List[Dict]:
        """Get recent conversations for context."""
        return self.context["conversations"][-limit:]
    
    def get_all_tasks(self) -> List[Dict]:
        """Get all tasks."""
        return self.context["tasks"]
    
    def get_all_facts(self) -> List[Dict]:
        """Get all facts."""
        return self.context["facts"]
    
    def clear(self):
        """Clear all memory."""
        self.context = self._create_default_memory()
        self.save()

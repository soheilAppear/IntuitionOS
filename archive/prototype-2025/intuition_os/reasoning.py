"""
Reasoning engine for IntuitionOS - the core brain using LLM.
"""
import os
from typing import Dict, Optional
from openai import OpenAI


class ReasoningEngine:
    """LLM-based reasoning engine that thinks aloud before acting."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize the reasoning engine with OpenAI API."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
    def think_and_act(self, user_input: str, context: Dict) -> Dict[str, str]:
        """
        Process user input through reasoning loop.
        Returns dict with 'reasoning' and 'action' keys.
        """
        if not self.client:
            return self._fallback_reasoning(user_input)
        
        # Build context from memory
        context_str = self._build_context(context)
        
        # System prompt that encourages thinking aloud
        system_prompt = """You are IntuitionOS, an operating system guided by intuition-based reasoning rather than pure logic.

You should:
1. Think through the user's request step-by-step (show your reasoning)
2. Consider the context and past interactions
3. Decide on the best action to take
4. Explain your reasoning before acting

Format your response as:
REASONING: [Your step-by-step thinking process]
ACTION: [The action you will take or response you will give]"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_str}\n\nUser Input: {user_input}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            return self._parse_response(content)
            
        except Exception as e:
            return {
                "reasoning": f"Error in reasoning engine: {str(e)}",
                "action": "I encountered an error. Please check your API configuration."
            }
    
    def _build_context(self, context: Dict) -> str:
        """Build context string from memory."""
        parts = []
        
        if context.get("conversations"):
            recent = context["conversations"][-3:]
            parts.append("Recent conversations:")
            for conv in recent:
                parts.append(f"- User: {conv['user_input']}")
                parts.append(f"  Response: {conv['response']}")
        
        if context.get("tasks"):
            parts.append("\nCurrent tasks:")
            for task in context["tasks"][-5:]:
                parts.append(f"- {task['task']} ({task['status']})")
        
        if context.get("facts"):
            parts.append("\nKnown facts:")
            for fact in context["facts"][-5:]:
                parts.append(f"- {fact['fact']}")
        
        return "\n".join(parts) if parts else "No previous context."
    
    def _parse_response(self, content: str) -> Dict[str, str]:
        """Parse the LLM response into reasoning and action."""
        lines = content.strip().split("\n")
        reasoning = []
        action = []
        current_section = None
        
        for line in lines:
            if line.startswith("REASONING:"):
                current_section = "reasoning"
                reasoning.append(line.replace("REASONING:", "").strip())
            elif line.startswith("ACTION:"):
                current_section = "action"
                action.append(line.replace("ACTION:", "").strip())
            elif current_section == "reasoning":
                reasoning.append(line)
            elif current_section == "action":
                action.append(line)
        
        return {
            "reasoning": "\n".join(reasoning).strip() or "Processing your request...",
            "action": "\n".join(action).strip() or content
        }
    
    def _fallback_reasoning(self, user_input: str) -> Dict[str, str]:
        """Fallback reasoning when API is not available."""
        return {
            "reasoning": "API key not configured. Using fallback mode. Set OPENAI_API_KEY to enable full reasoning.",
            "action": f"I received your input: '{user_input}'. Please configure the OpenAI API key to enable intelligent reasoning."
        }

"""
Task executor for IntuitionOS - handles command execution.
"""
import os
import subprocess
from typing import Dict, Optional


class TaskExecutor:
    """Executes tasks based on reasoning decisions."""
    
    SAFE_COMMANDS = {
        "ls": lambda: self._run_command(["ls", "-la"]),
        "pwd": lambda: self._run_command(["pwd"]),
        "date": lambda: self._run_command(["date"]),
        "whoami": lambda: self._run_command(["whoami"]),
        "echo": lambda args: self._run_command(["echo", args]),
    }
    
    def execute(self, action: str) -> str:
        """
        Execute an action based on the reasoning decision.
        For safety, only executes whitelisted commands.
        """
        action_lower = action.lower().strip()
        
        # Check for explicit commands
        if action_lower.startswith("run:"):
            command = action_lower.replace("run:", "").strip()
            return self._execute_safe_command(command)
        
        # Otherwise, return the action as a response
        return action
    
    def _execute_safe_command(self, command: str) -> str:
        """Execute a safe, whitelisted command."""
        cmd_parts = command.split(maxsplit=1)
        cmd_name = cmd_parts[0]
        cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""
        
        if cmd_name in self.SAFE_COMMANDS:
            try:
                if cmd_args:
                    result = self._run_command([cmd_name, cmd_args])
                else:
                    result = self._run_command([cmd_name])
                return f"Command output:\n{result}"
            except Exception as e:
                return f"Error executing command: {str(e)}"
        else:
            return f"Command '{cmd_name}' not in safe command list. Available: {', '.join(self.SAFE_COMMANDS.keys())}"
    
    def _run_command(self, cmd: list) -> str:
        """Run a system command and return output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return "Command timed out"
        except Exception as e:
            return f"Error: {str(e)}"

# LLM client with Ollama backend and a safe fallback

import os, requests

class LLMClient:
    def __init__(self, backend:str, model:str, temperature:float, max_tokens:int):
        # Save fields
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages):
        # Dispatch by backend
        if self.backend == "ollama":
            return self._chat_ollama(messages)
        # Fallback returns a trivial echo
        return "Ok. How can I help?"

    def _chat_ollama(self, messages):
        # Build endpoint from env or default
        base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        url = f"{base}/api/chat"
        # Compose payload
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            },
            "stream": False
        }
        try:
            # Post with timeout
            resp = requests.post(url, json=payload, timeout=60)
            # Raise for HTTP errors
            resp.raise_for_status()
            data = resp.json()
            # Return the assistant content if available
            return data.get("message", {}).get("content", "").strip() or "Ok."
        except Exception as e:
            # On error, fall back to a short string
            return f"(local model error: {e})"

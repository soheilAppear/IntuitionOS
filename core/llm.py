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
        base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        url = f"{base}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "").strip()
            return content or "Ok."
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach Ollama at {base} — run: ollama serve"
            )
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else "?"
            raise RuntimeError(
                f"Ollama returned HTTP {code} for model '{self.model}' — "
                f"is the model pulled? Run: ollama pull {self.model}"
            )
        except Exception as e:
            raise RuntimeError(f"LLM error: {e}")

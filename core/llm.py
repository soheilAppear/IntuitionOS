# LLM client with Ollama backend and an explicit fallback backend.

import json
import os

import requests


class LLMError(RuntimeError):
    """A failure to get a reply, distinguishable from a reply that says "error".

    Appendix A #13: the original client returned the error text as if it were
    model output, so a caller could not tell "the model said this" from "there
    was no model". A tool loop cannot tolerate that ambiguity — it would happily
    try to parse a connection error as a tool call.
    """


class LLMClient:
    def __init__(self, backend: str, model: str, temperature: float, max_tokens: int,
                 connect_timeout: float = 5.0, read_timeout: float = 120.0):
        # Save fields
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Split connect from read (Appendix A #14): a machine with no Ollama at
        # all should fail in seconds, while a 20B model on consumer hardware is
        # allowed to think for as long as it needs.
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

    def chat(self, messages, on_token=None):
        """Return the assistant's reply.

        Pass `on_token` to stream: it is called with each chunk as it arrives and
        the full text is still returned. With a tool loop in the picture a user
        otherwise sits in front of a still HUD for several seconds per iteration
        with no evidence anything is happening.
        """
        if self.backend == "ollama":
            return self._chat_ollama(messages, on_token=on_token)
        # Fallback returns a trivial echo
        return "Ok. How can I help?"

    def chat_json(self, messages, on_token=None):
        """Generate a JSON tool-loop turn without changing other chat callers.

        The tool protocol already asks for JSON in the system prompt. Ollama's
        JSON constraint keeps tool arguments parseable, and disabling optional
        thinking leaves the output-token budget available for those arguments.
        Ordinary chat (for example consolidation) keeps its existing format.
        """
        if self.backend == "ollama":
            return self._chat_ollama(messages, on_token=on_token, json_mode=True)
        return self.chat(messages, on_token=on_token)

    def _payload(self, messages, stream, *, json_mode=False):
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            "stream": stream,
        }
        if json_mode:
            payload.update(format="json", think=False)
        return payload

    def _chat_ollama(self, messages, on_token=None, *, json_mode=False):
        base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        url = f"{base}/api/chat"
        timeout = (self.connect_timeout, self.read_timeout)
        try:
            if on_token is None:
                resp = requests.post(url, json=self._payload(messages, False, json_mode=json_mode), timeout=timeout)
                resp.raise_for_status()
                result = resp.json()
                content = (result.get("message") or {}).get("content", "")
                return self._completed_content(content, result)
            return self._stream_ollama(url, messages, timeout, on_token, json_mode=json_mode)
        except requests.exceptions.ConnectionError:
            raise LLMError(f"Cannot reach Ollama at {base} — run: ollama serve")
        except requests.exceptions.ReadTimeout:
            raise LLMError(f"Ollama did not answer within {self.read_timeout:.0f}s for '{self.model}'")
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            raise LLMError(
                f"Ollama returned HTTP {code} for model '{self.model}' — "
                f"is the model pulled? Run: ollama pull {self.model}"
            )
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"LLM error: {e}")

    def _completed_content(self, content, final):
        if final.get("error"):
            raise LLMError(str(final["error"]))
        if final.get("done_reason") == "length":
            raise LLMError(
                f"Ollama reached the {self.max_tokens}-token output limit before finishing. "
                "Try a smaller request or increase max_tokens in config/config.yaml."
            )
        if final.get("done") is not True:
            raise LLMError("Ollama stopped before completing its reply. Please try again.")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Ollama completed without a visible reply. Please try again.")
        return content.strip()

    def _stream_ollama(self, url, messages, timeout, on_token, *, json_mode=False):
        parts = []
        final = {}
        with requests.post(url, json=self._payload(messages, True, json_mode=json_mode), timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    raise LLMError("Ollama returned an invalid streaming frame. Please try again.")
                if not isinstance(chunk, dict):
                    raise LLMError("Ollama returned an invalid streaming frame. Please try again.")
                if chunk.get("error"):
                    raise LLMError(str(chunk["error"]))
                piece = (chunk.get("message") or {}).get("content", "")
                if piece:
                    if not isinstance(piece, str):
                        raise LLMError("Ollama returned invalid reply content. Please try again.")
                    parts.append(piece)
                    try:
                        on_token(piece)
                    except Exception:
                        # A UI that has gone away must not kill the generation.
                        pass
                if chunk.get("done"):
                    final = chunk
                    break
        return self._completed_content("".join(parts), final)

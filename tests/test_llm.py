"""Ollama protocol boundaries: incomplete output must never become a tool call."""

import json

import pytest
import requests

from core.llm import LLMClient, LLMError


class Response:
    def __init__(self, body=None, frames=()):
        self.body = body
        self.frames = frames

    def raise_for_status(self):
        pass

    def json(self):
        return self.body

    def iter_lines(self):
        for frame in self.frames:
            yield frame if isinstance(frame, bytes) else json.dumps(frame).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def client():
    return LLMClient("ollama", "test-model", 0.2, 600)


@pytest.fixture
def http(monkeypatch):
    calls = []

    def install(response):
        def post(url, **kwargs):
            calls.append((url, kwargs))
            return response
        monkeypatch.setattr("core.llm.requests.post", post)
        return calls
    return install


def finished(content='{"reply":"Ready"}', **kwargs):
    return {"message": {"content": content}, "done": True, "done_reason": "stop", **kwargs}


@pytest.mark.parametrize("stream", [False, True])
def test_json_generation_preserves_messages_and_requests_json_without_thinking(client, http, stream):
    message = [{"role": "user", "content": "Create a Python file"}]
    frames = [{"message": {"content": '{"reply":'}}, finished('"Ready"}')]
    calls = http(Response(body=finished(), frames=frames))
    tokens = []
    assert client.chat_json(message, on_token=tokens.append if stream else None) == '{"reply":"Ready"}'
    payload = calls[0][1]["json"]
    assert payload["format"] == "json" and payload["think"] is False
    assert payload["messages"] == message and payload["model"] == "test-model"
    assert payload["options"]["num_predict"] == 600
    assert payload["stream"] is stream
    assert calls[0][1]["timeout"] == (5.0, 120.0)
    if stream:
        assert "".join(tokens) == '{"reply":"Ready"}'


@pytest.mark.parametrize("stream", [False, True])
def test_ordinary_chat_does_not_inherit_json_or_thinking_options(client, http, stream):
    calls = http(Response(body=finished("some prose"), frames=[finished("some prose")]))
    assert client.chat([], on_token=(lambda piece: None) if stream else None) == "some prose"
    assert "format" not in calls[0][1]["json"]
    assert "think" not in calls[0][1]["json"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("content", ["", "  \n", None])
def test_empty_or_thinking_only_result_is_an_error_not_fabricated_ok(client, http, stream, content):
    result = finished(content)
    result["message"]["thinking"] = "Thinking without a visible answer"
    http(Response(body=result, frames=[result]))
    with pytest.raises(LLMError, match="without a visible reply"):
        client.chat_json([], on_token=(lambda piece: None) if stream else None)


@pytest.mark.parametrize("stream", [False, True])
def test_length_stop_rejects_even_apparently_complete_json(client, http, stream):
    result = finished('{"tool":"write_file","args":{"path":"x.py","text":"partial"}}', done_reason="length")
    http(Response(body=result, frames=[result]))
    with pytest.raises(LLMError, match="600-token output limit"):
        client.chat_json([], on_token=(lambda piece: None) if stream else None)


@pytest.mark.parametrize("stream", [False, True])
def test_incomplete_response_is_never_accepted(client, http, stream):
    result = {"message": {"content": '{"reply":"Looks valid"}'}, "done": False}
    http(Response(body=result, frames=[result]))
    with pytest.raises(LLMError, match="before completing"):
        client.chat_json([], on_token=(lambda piece: None) if stream else None)


@pytest.mark.parametrize("frame", [b"broken json", [], {"message": {"content": ["bad"]}}])
def test_malformed_stream_frames_cannot_be_silently_lost(client, http, frame):
    http(Response(frames=[frame, finished()]))
    with pytest.raises(LLMError, match="invalid"):
        client.chat_json([], on_token=lambda piece: None)


def test_callback_failure_does_not_destroy_complete_answer(client, http):
    http(Response(frames=[b"", finished()]))
    def disconnected(piece):
        raise RuntimeError("HUD closed")
    assert client.chat_json([], on_token=disconnected) == '{"reply":"Ready"}'


@pytest.mark.parametrize("stream", [False, True])
def test_provider_error_is_reported(client, http, stream):
    result = {"error": "model unavailable"}
    http(Response(body=result, frames=[result]))
    with pytest.raises(LLMError, match="model unavailable"):
        client.chat_json([], on_token=(lambda piece: None) if stream else None)


@pytest.mark.parametrize("error, match", [
    (requests.exceptions.ConnectionError(), "Cannot reach Ollama"),
    (requests.exceptions.ReadTimeout(), "did not answer within"),
])
def test_transport_failures_remain_errors(client, monkeypatch, error, match):
    def failed(*args, **kwargs):
        raise error
    monkeypatch.setattr("core.llm.requests.post", failed)
    with pytest.raises(LLMError, match=match):
        client.chat_json([])

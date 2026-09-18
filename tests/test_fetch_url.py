"""Reading a page, and refusing to read the wrong things.

`open_url` shows a page to the user and can never hand its contents back, so a
question like "how is the weather" had no tool that could answer it. `fetch_url`
is that tool, and the interesting part is everything it declines to fetch.
"""

import pytest

from core import os_sandbox
from core.actions import register_os_capabilities
from core.capabilities import capabilities, gate


@pytest.fixture(autouse=True)
def _registered():
    register_os_capabilities()


# ── What it refuses ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:11434/api/tags",   # Ollama's own API
    "http://localhost:7432/health",      # the IntuitionOS backend
    "http://192.168.1.1",                # the router
    "http://10.0.0.5/admin",             # private network
    "http://169.254.169.254/latest/",    # cloud metadata
])
def test_addresses_inside_this_machine_or_network_are_refused(url, monkeypatch):
    """Without this the model can read every local service the user is running."""
    def _never_called(*a, **kw):
        raise AssertionError("a request was made to a private address")
    monkeypatch.setattr(os_sandbox, "fetch_url", os_sandbox.fetch_url)

    result = os_sandbox.fetch_url(url)
    assert "error" in result
    assert "private network" in result["error"] or "this machine" in result["error"]


@pytest.mark.parametrize("url", [
    "file:///C:/Windows/win.ini",
    "ftp://example.com/x",
    "javascript:alert(1)",
    "",
])
def test_only_http_and_https_are_accepted(url):
    assert "error" in os_sandbox.fetch_url(url)


def test_a_public_name_that_resolves_to_loopback_is_refused(monkeypatch):
    """The DNS-rebinding shape: a public-looking name pointing home."""
    import socket
    monkeypatch.setattr(
        os_sandbox.socket, "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, None, None, "", ("127.0.0.1", 80))],
    )
    result = os_sandbox.fetch_url("https://totally-public.example")
    assert "error" in result and "127.0.0.1" in result["error"]


# ── What it returns ─────────────────────────────────────────────────────────


def test_html_is_reduced_to_readable_text():
    html = """
      <html><head><title>T</title><style>body{color:red}</style>
      <script>var x = 1 < 2;</script></head>
      <body><h1>Weather</h1><p>It is 18&deg;C and raining.</p></body></html>
    """
    text = os_sandbox.html_to_text(html)
    assert "It is 18°C and raining." in text
    assert "var x" not in text, "script contents must not reach the model"
    assert "color:red" not in text, "stylesheet contents must not reach the model"
    assert "<" not in text and ">" not in text


def test_the_returned_text_is_capped(monkeypatch):
    class Response:
        status_code, ok, encoding = 200, True, "utf-8"
        url = "https://example.invalid/"
        headers = {"Content-Type": "text/html"}
        class raw:
            @staticmethod
            def read(_n, decode_content=True):
                return b"<p>" + b"long " * 5000 + b"</p>"
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(os_sandbox, "_refuse_internal_address", lambda _h: None)
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **kw: Response())

    result = os_sandbox.fetch_url("https://example.invalid", max_chars=500)
    assert result["ok"] and result["truncated"]
    assert len(result["text"]) < 600


# ── How the gate treats it ──────────────────────────────────────────────────


def test_the_anticipator_may_never_fetch_a_url():
    """`free` is what the anticipator may run speculatively. A network request
    to a URL it merely guessed at must never be one of those."""
    cap = capabilities.get("os_fetch_url")
    assert cap.reversibility != "free"

    decision = gate(cap, {"url": "https://example.com"},
                    actor="anticipator", confidence=1.0)
    assert decision.verdict == "deny"


def test_the_model_must_ask_before_opening_a_browser_but_the_user_need_not():
    """Opening a window because the user asked is the request; opening one
    because a model inferred it is a surprise on their screen."""
    args = {"url": "https://example.com", "browser": "chrome"}

    cap = capabilities.get("os_open_url")
    assert gate(cap, dict(args), actor="model", confidence=1.0).verdict == "confirm"
    assert gate(cap, dict(args), actor="user", confidence=1.0).verdict == "allow"

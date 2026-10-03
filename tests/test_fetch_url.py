"""Reading a page, and refusing to read the wrong things.

`open_url` shows a page to the user and can never hand its contents back, so a
question like "how is the weather" had no tool that could answer it. `fetch_url`
is that tool, and the interesting part is everything it declines to fetch.
"""

from types import SimpleNamespace

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


class _Response:
    def __init__(self, body=b"Boston: sunny, +18 C", *, status=200,
                 content_type="text/plain; charset=utf-8", location=None,
                 encoding="utf-8"):
        self.body = body
        self.status_code = status
        self.ok = status < 400
        self.encoding = encoding
        self.url = "https://wttr.in/Boston?format=3"
        self.headers = {"Content-Type": content_type}
        if location is not None:
            self.headers["Location"] = location
        self.reads = 0
        self.closed = False
        self.raw = SimpleNamespace(read=self.read)

    def read(self, size, decode_content=True):
        self.reads += 1
        return self.body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


@pytest.fixture
def fetch_responses(monkeypatch):
    import requests

    def install(*responses):
        calls = []
        pending = iter(responses)

        def get(url, **kwargs):
            calls.append((url, kwargs))
            response = next(pending)
            response.url = url
            return response

        monkeypatch.setattr(os_sandbox, "_refuse_internal_address", lambda host: None)
        monkeypatch.setattr(requests, "get", get)
        return calls

    return install


def test_weather_text_is_returned_without_losing_unicode(fetch_responses):
    response = _Response("Boston: ☀️ +18°C".encode())
    fetch_responses(response)
    result = os_sandbox.fetch_url("https://wttr.in/Boston?format=3")
    assert result["ok"] and not result["truncated"]
    assert result["text"] == "Boston: ☀️ +18°C"
    assert response.closed


@pytest.mark.parametrize("content_type", ["image/png", "application/pdf", "application/octet-stream"])
def test_binary_content_is_rejected_before_reading(fetch_responses, content_type):
    response = _Response(b"binary data", content_type=content_type)
    fetch_responses(response)
    result = os_sandbox.fetch_url("https://example.com/file")
    assert "not a readable page" in result["error"]
    assert response.reads == 0
    assert response.closed


@pytest.mark.parametrize("content_type", ["", "application/json", "application/ld+json", "application/xml"])
def test_text_data_formats_remain_readable(fetch_responses, content_type):
    fetch_responses(_Response(b"readable payload", content_type=content_type))
    assert os_sandbox.fetch_url("https://example.com")["text"] == "readable payload"


def test_http_errors_do_not_read_an_error_body(fetch_responses):
    response = _Response(status=503)
    fetch_responses(response)
    result = os_sandbox.fetch_url("https://example.com")
    assert result["status_code"] == 503
    assert "HTTP 503" in result["error"]
    assert response.reads == 0
    assert response.closed


def test_unknown_charset_falls_back_to_utf8(fetch_responses):
    fetch_responses(_Response("18°C".encode(), encoding="not-a-real-charset"))
    assert os_sandbox.fetch_url("https://example.com")["text"] == "18°C"


def test_byte_limit_is_reported_even_if_html_removal_shortens_text(fetch_responses, monkeypatch):
    monkeypatch.setattr(os_sandbox, "_FETCH_MAX_BYTES", 100)
    fetch_responses(_Response(b"<p>" + b" " * 200 + b"rest</p>", content_type="text/html"))
    result = os_sandbox.fetch_url("https://example.com")
    assert result["ok"] and result["truncated"]
    assert result["text"].endswith("… (truncated)")


def test_relative_public_redirect_is_followed_and_closed(fetch_responses):
    redirect = _Response(status=302, location="/Boston?format=3")
    response = _Response()
    calls = fetch_responses(redirect, response)
    result = os_sandbox.fetch_url("https://wttr.in")
    assert result["text"] == "Boston: sunny, +18 C"
    assert result["url"] == "https://wttr.in/Boston?format=3"
    assert all(call[1]["allow_redirects"] is False for call in calls)
    assert redirect.closed and response.closed


def test_redirect_to_private_address_is_refused_before_second_request(fetch_responses, monkeypatch):
    calls = fetch_responses(_Response(status=302, location="http://127.0.0.1:11434/api/tags"))
    checked = []

    def refuse(host):
        checked.append(host)
        return "Refusing private network destination" if host == "127.0.0.1" else None

    monkeypatch.setattr(os_sandbox, "_refuse_internal_address", refuse)
    result = os_sandbox.fetch_url("https://example.com")
    assert "private network" in result["error"]
    assert checked == ["example.com", "127.0.0.1"]
    assert len(calls) == 1


@pytest.mark.parametrize("location", ["file:///C:/Windows/win.ini", "ftp://example.com"])
def test_redirect_to_non_http_scheme_is_rejected(fetch_responses, location):
    calls = fetch_responses(_Response(status=302, location=location))
    assert "error" in os_sandbox.fetch_url("https://example.com")
    assert len(calls) == 1


def test_redirect_loop_is_bounded(fetch_responses):
    responses = [_Response(status=302, location="/again")
                 for _ in range(os_sandbox._FETCH_MAX_REDIRECTS + 1)]
    calls = fetch_responses(*responses)
    result = os_sandbox.fetch_url("https://example.com")
    assert "too many redirects" in result["error"]
    assert len(calls) == os_sandbox._FETCH_MAX_REDIRECTS + 1
    assert all(response.closed for response in responses)


def test_fetch_timeout_is_returned_as_an_error(monkeypatch):
    import requests

    def timeout(*args, **kwargs):
        raise requests.Timeout("weather service timed out")

    monkeypatch.setattr(os_sandbox, "_refuse_internal_address", lambda host: None)
    monkeypatch.setattr(requests, "get", timeout)
    result = os_sandbox.fetch_url("https://wttr.in/Boston?format=3")
    assert "timed out" in result["error"]


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

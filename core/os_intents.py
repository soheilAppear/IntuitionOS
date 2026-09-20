"""Local phrase recognition shared by command resolution and the interfaces.

is_app_command protects established app phrases from shell-name correction.
_try_os_intent returns an action name and arguments for the capability registry;
matching a phrase neither executes that action nor grants permission for it.
Rule order is intentional because some natural-language patterns overlap.
"""

import re

from core.file_intents import is_file_request, try_file_intent


def is_app_command(text):
    """Return whether bare input belongs to the existing application grammar.

    A prefix check keeps shell arguments such as ``git commit -m battery`` from
    being treated as OS requests. Explicit /exec routing is handled by callers.
    """
    stripped = text.strip()
    if is_file_request(stripped) or _is_natural_action_request(stripped):
        return True
    if stripped in {"ls", "tree"} or re.match(
        r"^(?:remind\s+me|read\s+file)\s+", stripped, re.I
    ):
        return True
    if _try_browser_intent(stripped) is not None or _looks_like_browser_request(stripped):
        return True
    # Phrase patterns search within sentences; restrict the initial word before
    # applying them to preserve command heads and their opaque arguments.
    prefix = stripped.lower().split(maxsplit=1)[0] if stripped else ""
    natural_prefixes = {
        "open",
        "launch",
        "start",
        "run",
        "execute",
        "set",
        "put",
        "change",
        "turn",
        "make",
        "adjust",
        "volume",
        "sound",
        "mute",
        "no",
        "silent",
        "silence",
        "raise",
        "increase",
        "lower",
        "decrease",
        "what",
        "what's",
        "check",
        "get",
        "show",
        "brightness",
        "dim",
        "battery",
        "charge",
        "power",
        "how",
        "am",
        "which",
        "network",
        "ip",
        "wifi",
        "wi-fi",
        "enable",
        "disable",
        "sleep",
        "hibernate",
        "suspend",
        "lock",
        "shutdown",
        "shut",
        "restart",
        "reboot",
        "cancel",
        "screenshot",
        "screen",
        "snap",
        "take",
        "system",
        "sys",
        "pc",
        "hardware",
        "computer",
        "machine",
        "windows",
        "resource",
        "performance",
        "ram",
        "memory",
        "cpu",
        "processor",
        "disk",
        "storage",
        "tell",
        "give",
        "list",
        "display",
        "running",
        "kill",
        "terminate",
        "close",
        "stop",
        "quit",
        "end",
        "force",
        "read",
        "clipboard",
    }
    return (
        prefix in natural_prefixes or prefix in _KNOWN_APP_NAMES
    ) and _try_os_intent(text) is not None


_VOL_WORDS = {
    "zero": 0,
    "muted": 0,
    "ten": 10,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "half": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "full": 100,
    "max": 100,
    "maximum": 100,
    "twenty five": 25,
    "seventy five": 75,
}

# App names that should always route to os_open_app
_KNOWN_APP_NAMES = {
    "chrome",
    "google chrome",
    "google",
    "firefox",
    "edge",
    "browser",
    "web browser",
    "my browser",
    "vscode",
    "vs code",
    "visual studio code",
    "code",
    "notepad",
    "calculator",
    "calc",
    "explorer",
    "file explorer",
    "terminal",
    "cmd",
    "spotify",
    "discord",
    "slack",
    "teams",
    "word",
    "excel",
    "powerpoint",
    "paint",
    "task manager",
    "steam",
    "obs",
    "vlc",
    "settings",
    "control panel",
}

_APP_NAME_ALIAS = {
    "google chrome": "chrome",
    "google": "chrome",
    "browser": "chrome",
    "web browser": "chrome",
    "my browser": "chrome",
    "vs code": "vscode",
    "visual studio code": "vscode",
    "code editor": "vscode",
    "windows terminal": "terminal",
    "command prompt": "terminal",
    "cmd": "terminal",
    "file manager": "explorer",
    "file explorer": "explorer",
    "files": "explorer",
    "calc": "calculator",
}


_BROWSER_PATTERN = (
    r"(?P<browser>google\s+chrome|cvhrome|chrome|microsoft\s+edge|edge|"
    r"mozilla\s+firefox|firefox|(?:default\s+|web\s+|my\s+)?browser)"
)
_BROWSER_ALIASES = {
    "google chrome": "chrome", "cvhrome": "chrome",
    "microsoft edge": "edge", "mozilla firefox": "firefox",
    "browser": "default", "default browser": "default",
    "web browser": "default", "my browser": "default",
}
_NAVIGATE_PATTERN = r"(?:go\s+to|goto(?:\s+to)?|navigate\s+to|browse\s+to|open|visit)"
_BROWSER_CONNECTOR = r"\s*,?\s+(?:(?:and(?:\s+then)?|then)\s+)?"
_BROWSER_REQUEST_PATTERNS = (
    rf"(?:open|launch|start)\s*(?:the\s+|my\s+)?{_BROWSER_PATTERN}(?:\s+browser)?"
    rf"(?:\s+for\s+me)?{_BROWSER_CONNECTOR}{_NAVIGATE_PATTERN}\s+(?P<url>\S+)",
    rf"(?:go\s+to|goto(?:\s+to)?)\s+(?:the\s+|my\s+)?{_BROWSER_PATTERN}"
    rf"\s*,?\s+(?:and\s+)?open\s+it(?:\s+for\s+me)?"
    rf"{_BROWSER_CONNECTOR}{_NAVIGATE_PATTERN}\s+(?P<url>\S+)",
    rf"(?:go\s+to|goto(?:\s+to)?)\s+(?:the\s+|my\s+)?{_BROWSER_PATTERN}"
    rf"{_BROWSER_CONNECTOR}{_NAVIGATE_PATTERN}\s+(?P<url>\S+)",
    rf"{_NAVIGATE_PATTERN}\s+(?P<url>\S+)"
    rf"(?:\s+(?:in|with|using)\s+(?:the\s+|my\s+)?{_BROWSER_PATTERN})?",
)


def _browser_request_text(text: str) -> str:
    return re.sub(
        r"^(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?", "", text.strip(),
        flags=re.I,
    )


def _try_browser_intent(text: str):
    """Recognize complete website requests before generic app-name matching."""
    original = _browser_request_text(text)
    for pattern in _BROWSER_REQUEST_PATTERNS:
        match = re.fullmatch(pattern + r"(?:\s+for\s+me)?(?:\s+please)?[.!]?", original, re.I)
        if not match:
            continue
        raw_url = match.group("url")
        # Sentence punctuation on a bare hostname is unambiguous; punctuation
        # inside a URL path or query is data and must survive unchanged.
        if re.fullmatch(r"(?:https?://)?[^/?#]+[.!?]", raw_url, re.I):
            raw_url = raw_url.rstrip(".!?")
        browser = match.groupdict().get("browser") or "default"
        browser = " ".join(browser.lower().split())
        browser = _BROWSER_ALIASES.get(browser, browser)
        # A bare app/executable or local file remains in the existing app grammar.
        # Explicit browser navigation also supports single-label intranet hosts.
        if pattern == _BROWSER_REQUEST_PATTERNS[-1] and browser == "default" and "://" not in raw_url:
            host = raw_url.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
            if (
                "." not in host and not re.fullmatch(r"localhost(?::\d+)?|\[.+\](?::\d+)?", host, re.I)
            ) or re.search(r"\.(?:exe|bat|cmd|ps1|py|txt|md|json|pdf|docx|xlsx)$", host, re.I):
                continue
        from core.os_sandbox import normalize_url
        try:
            url = normalize_url(raw_url)
        except ValueError:
            continue
        return ("os_open_url", {"url": url, "browser": browser})
    return None


def _looks_like_browser_request(text: str) -> bool:
    """Keep unsupported compound browser prose out of shell/partial OS routing."""
    original = _browser_request_text(text)
    # A browser task may describe the next action without naming a URL yet:
    # "open the chrome and check the weather for me". Preserve the entire
    # instruction for the model instead of correcting 'open' or launching only
    # the browser. Requiring a conjunction leaves executable flags opaque.
    browser_start = (
        rf"(?:(?:open|launch|start)\s*|(?:go\s+to|goto(?:\s+to)?)\s+)"
        rf"(?:the\s+|my\s+)?{_BROWSER_PATTERN}(?:\s+browser)?"
        rf"(?:\s+for\s+me)?\s*,?\s+(?:and(?:\s+then)?|then)\s+\S"
    )
    if re.match(browser_start, original, re.I):
        return True
    for pattern in _BROWSER_REQUEST_PATTERNS:
        match = re.match(pattern + r"(?=\s|$)", original, re.I)
        if match and _try_browser_intent(original[:match.end()]) is not None:
            return True
    return False


def _is_natural_action_request(text: str) -> bool:
    """Protect grammatical requests without guessing the requested action.

    Determiners and question lead-ins distinguish prose from executable flags,
    paths, and bare arguments. Explicit /exec continues to select shell routing.
    This only classifies the input; it neither executes nor rewrites anything.
    """
    original = _browser_request_text(text)
    return re.match(
        r"(?:check|show|find|tell)\s+(?:me\s+)?(?:about\s+)?"
        r"(?:the|my|our|your|this|that|these|those|a|an|what|how|where|when|whether|if)\s+\S",
        original,
        re.I,
    ) is not None


# ── Windows on the screen ────────────────────────────────────────────────────
#
# Said aloud, arranging windows is the most natural thing to ask for and the
# least worth waking a model over: "put this on the left" is unambiguous, and
# routing it here means it works with Ollama stopped and answers immediately.

_THIS_WINDOW = r"(?:(?:this|the|current|active|it)\s+)?(?:window|one)?"
_POSITIONS = {
    "left": "left", "right": "right", "top": "top", "bottom": "bottom",
    "up": "top", "down": "bottom",
    "top left": "top-left", "top right": "top-right",
    "bottom left": "bottom-left", "bottom right": "bottom-right",
    "upper left": "top-left", "upper right": "top-right",
    "lower left": "bottom-left", "lower right": "bottom-right",
    "full": "full", "fullscreen": "full", "full screen": "full",
}


def _try_window_intent(t: str):
    """Recognise window arrangement without interpreting it as a shell command."""
    t = _browser_request_text(t).strip().rstrip(".!?")

    # "snap/move/put this window to the left", "snap left"
    match = re.fullmatch(
        r"(?:snap|move|put|send|dock|place)\s+" + _THIS_WINDOW +
        r"\s*(?:to|on|at|into)?\s*(?:the\s+)?(?P<where>[a-z ]+?)"
        r"(?:\s+(?:half|side|corner|of\s+the\s+screen))?",
        t,
    )
    if match:
        where = " ".join(match.group("where").split())
        position = _POSITIONS.get(where)
        if position:
            return ("os_snap_window", {"position": position})

    match = re.fullmatch(
        r"(?P<state>maximi[sz]e|minimi[sz]e|restore)\s*" + _THIS_WINDOW, t)
    if match:
        state = match.group("state")
        canonical = ("maximize" if state.startswith("maxim")
                     else "minimize" if state.startswith("minim") else "restore")
        return ("os_window_state", {"state": canonical})

    match = re.fullmatch(
        r"(?:(?:switch|go|move|cycle)\s+to\s+(?:the\s+)?)?"
        r"(?P<direction>next|previous|last|prior)\s+(?:window|app|application)", t)
    if match:
        direction = match.group("direction")
        return ("os_cycle_window",
                {"direction": "next" if direction == "next" else "previous"})

    if re.fullmatch(r"(?:list|show|what)\s+(?:my\s+|the\s+|are\s+my\s+)?"
                    r"(?:open\s+)?windows(?:\s+are\s+open)?", t):
        return ("os_list_windows", {})

    return None


def _try_os_intent(text: str):
    """Return (capability_name, argument_dict) for the first matching OS phrase.

    Unrecognized input returns None. Matching normalizes case and common aliases;
    validation and execution remain responsibilities of the capability gate.
    """
    file_intent = try_file_intent(text)
    if file_intent is not None:
        return file_intent
    if is_file_request(text):
        return None

    browser_intent = _try_browser_intent(text)
    if browser_intent is not None:
        return browser_intent
    if _looks_like_browser_request(text):
        return None
    if _is_natural_action_request(text) and re.search(r"\b(?:and|then)\s+\S", text, re.I):
        # The loose OS patterns below cannot fulfill a compound instruction.
        return None

    t = text.lower().strip()

    window_intent = _try_window_intent(t)
    if window_intent is not None:
        return window_intent

    # open / launch / start  (action-first: "open chrome") ────────────
    m = re.match(
        r"^(?:open|launch|start|run|execute)\s+(?:the\s+|my\s+|a\s+)?"
        r"(.+?)(?:\s+(?:app(?:lication)?|program|browser))?[.!?]?$",
        t,
    )
    if m:
        name = m.group(1).strip().rstrip(".,!?")
        name = _APP_NAME_ALIAS.get(name, name)
        words = name.split()
        if name in _KNOWN_APP_NAMES or (
            1 <= len(words) <= 3
            and not any(
                w in ("a", "an", "the", "new", "quick", "file", "search", "question")
                for w in words
            )
        ):
            return ("os_open_app", {"name": name})

    # open / launch  (name-first: "chrome open", "spotify launch") ─────
    m = re.match(r"^(.+?)\s+(?:open|launch|start|run)[.!?]?$", t)
    if m:
        name = m.group(1).strip().rstrip(".,!?")
        name = _APP_NAME_ALIAS.get(name, name)
        words = name.split()
        if name in _KNOWN_APP_NAMES or (1 <= len(words) <= 2):
            return ("os_open_app", {"name": name})

    # volume numeric — "set/put/change volume to/at/as X[%]" ──────────
    m = (
        re.search(
            r"(?:set|put|change|turn|make|adjust)\s+(?:the\s+)?(?:volume|sound)\s+(?:to|at|as|=)\s*(\d+)",
            t,
        )
        or re.search(r"\bvolume\s+(?:to\s+|at\s+|=\s*)?(\d+)\b", t)
        or re.search(r"\b(\d+)\s*%?\s*(?:volume|loudness)\b", t)
    )
    if m:
        return ("os_set_volume", {"level": int(m.group(1))})

    # volume word numbers ("set volume to fifty") ───────────────────────
    for phrase, num in _VOL_WORDS.items():
        pesc = re.escape(phrase)
        if (
            re.search(rf"(?:volume|sound)\s+(?:to\s+|at\s+|=\s*)?{pesc}\b", t)
            or re.search(
                rf"(?:set|put|change|make)\s+(?:the\s+)?(?:volume|sound)\s+(?:to\s+|at\s+)?{pesc}\b",
                t,
            )
            or re.search(rf"\b{pesc}\s*(?:percent\s+)?(?:volume|loudness)\b", t)
        ):
            return ("os_set_volume", {"level": num})

    # volume relative ───────────────────────────────────────────────────
    if re.search(r"\bmute\b|\bno\s+sound\b|\bsilent(?:ce)?\b", t):
        return ("os_set_volume", {"level": 0})
    if (
        re.search(r"(?:volume|sound)\s+(?:up|max|full|loud|higher|louder)", t)
        or re.search(r"turn\s+(?:up|the\s+volume\s+up|volume\s+up)", t)
        or re.search(r"turn\s+up\s+(?:the\s+)?(?:volume|sound)", t)
        or re.search(r"(?:raise|increase)\s+(?:the\s+)?(?:volume|sound)", t)
    ):
        return ("os_set_volume", {"level": 90})
    if (
        re.search(r"(?:volume|sound)\s+(?:down|low|quiet(?:er)?|half|lower)", t)
        or re.search(r"turn\s+(?:down|the\s+volume\s+down|volume\s+down)", t)
        or re.search(r"turn\s+(?:the\s+)?(?:volume|sound)\s+down", t)
        or re.search(r"(?:lower|decrease|reduce)\s+(?:the\s+)?(?:volume|sound)", t)
    ):
        return ("os_set_volume", {"level": 20})

    # get/check current volume ──────────────────────────────────────────
    if re.search(
        r"(?:what|check|get|show)\s+(?:is\s+)?(?:the\s+)?(?:current\s+)?(?:volume|sound\s+level)",
        t,
    ):
        return ("os_get_volume", {})

    # brightness ────────────────────────────────────────────────────────
    m = (
        re.search(
            r"(?:set|put|change|adjust)\s+(?:the\s+)?brightness\s+(?:to|at|=)\s*(\d+)",
            t,
        )
        or re.search(r"\bbrightness\s+(?:to\s+|at\s+)?(\d+)\b", t)
        or re.search(r"\b(\d+)\s*%?\s*brightness\b", t)
    )
    if m:
        return ("os_set_brightness", {"level": int(m.group(1))})
    if re.search(r"\bbrightness\s+(?:up|higher|brighter|max|full)\b", t) or re.search(
        r"(?:increase|raise|turn\s+up)\s+(?:the\s+)?brightness", t
    ):
        return ("os_set_brightness", {"level": 100})
    if re.search(r"\bbrightness\s+(?:down|lower|dim|half|low)\b", t) or re.search(
        r"(?:decrease|lower|dim|reduce|turn\s+down)\s+(?:the\s+)?brightness", t
    ):
        return ("os_set_brightness", {"level": 30})
    if re.search(
        r"(?:what|check|get|show)\s+(?:is\s+)?(?:the\s+)?(?:current\s+)?brightness", t
    ):
        return ("os_get_brightness", {})

    # battery ───────────────────────────────────────────────────────────
    if re.search(
        r"\b(?:battery|charge|power\s+level|how\s+much\s+(?:battery|charge|power)|"
        r"battery\s+(?:life|level|status|percentage|percent)|"
        r"how\s+long\s+(?:until|till|before).*(?:battery|dies|dead))\b",
        t,
    ):
        return ("os_get_battery", {})

    # network / wifi ────────────────────────────────────────────────────
    if re.search(
        r"\b(?:network\s+info(?:rmation)?|(?:what(?:\'s|\s+is)\s+(?:my|the)\s+)?(?:ip|wifi|wi-fi|ssid|"
        r"connection|internet)\s+(?:address|status|info|name)?|"
        r"am\s+i\s+connected|what\s+network|which\s+wifi|show\s+network)\b",
        t,
    ):
        return ("os_get_network_info", {})
    if re.search(r"\b(?:enable|turn\s+on)\s+(?:the\s+)?(?:wifi|wi-fi|wireless)\b", t):
        return ("os_toggle_wifi", {"state": "on"})
    if re.search(r"\b(?:disable|turn\s+off)\s+(?:the\s+)?(?:wifi|wi-fi|wireless)\b", t):
        return ("os_toggle_wifi", {"state": "off"})

    # sleep / lock ──────────────────────────────────────────────────────
    if (
        re.search(
            r"\b(?:sleep|hibernate|suspend)\s+(?:(?:the|my)\s+)?(?:computer|pc|machine|laptop)?\b",
            t,
        )
        and "wake" not in t
    ):
        return ("os_sleep_computer", {})
    if re.search(
        r"\b(?:lock\s+(?:(?:the|my)\s+)?(?:screen|computer|pc|machine|laptop)?|"
        r"(?:screen\s+)?lock)\b",
        t,
    ):
        return ("os_lock_screen", {})

    # power / shutdown / restart ────────────────────────────────────────
    if re.search(
        r"\b(?:shut\s*down|shutdown|power\s*off|turn\s+off)\s+(?:(?:the|my|this|your)\s+)?(?:computer|pc|machine|system|laptop)\b",
        t,
    ):
        return ("os_shutdown_computer", {"delay_sec": 30})
    if re.search(
        r"\b(?:restart|reboot)\s+(?:(?:the|my|this|your)\s+)?(?:computer|pc|machine|system|laptop)\b",
        t,
    ):
        return ("os_restart_computer", {"delay_sec": 30})
    if re.search(r"\bcancel\s+(?:the\s+)?(?:shutdown|restart|reboot)\b", t):
        return ("os_cancel_shutdown", {})

    # screenshot ────────────────────────────────────────────────────────
    if re.search(r"screenshot|screen\s+cap(?:ture)?|snap\s+(?:the\s+)?screen", t):
        return ("os_take_screenshot", {})

    # system info / resource report ────────────────────────────────────
    if re.search(
        r"\b(?:"
        # direct terms
        r"system\s+info(?:rmation)?|sys(?:tem)?\s+status|pc\s+info(?:rmation)?|"
        r"hardware\s+info(?:rmation)?|computer\s+stats?|machine\s+info|"
        # resource variants
        r"(?:windows|system|pc|computer|my)\s+resources?|"
        r"resource\s+(?:report|usage|status|info|check)|"
        r"performance\s+(?:report|status|info|stats?)|"
        # RAM / memory
        r"ram(?:\s+(?:usage|status|info|report|check))?|"
        r"memory\s+(?:usage|status|info|report|check|left)|"
        r"how\s+much\s+(?:ram|memory|storage|disk\s+space)|"
        # CPU
        r"cpu(?:\s+(?:usage|load|status|info|report|check|temp(?:erature)?))?|"
        r"processor\s+(?:usage|load|status|info)|"
        # disk
        r"disk\s+(?:space|usage|status|info)|storage\s+(?:space|status|info)|"
        # catch-alls
        r"tell\s+me\s+about\s+(?:my\s+)?(?:system|resources?|computer|pc)|"
        r"(?:show|give\s+me|check)\s+(?:(?:my|the)\s+)?(?:system|resource|performance|pc|computer)\s+(?:report|status|info|stats?|usage)"
        r")\b",
        t,
    ):
        return ("os_system_info", {})

    # processes ─────────────────────────────────────────────────────────
    if re.search(
        r"\b(?:(?:list|show|display|what(?:\'s|\s+is)?)\s+(?:running|"
        r"(?:all\s+)?processes?|(?:open\s+)?apps?|programs?)|"
        r"running\s+(?:processes?|apps?|programs?)|"
        r"what\s+(?:apps?|programs?)\s+(?:are\s+)?(?:open|running))\b",
        t,
    ):
        return ("os_list_processes", {})

    # kill process ──────────────────────────────────────────────────────
    m = re.match(
        r"^(?:kill|close|stop|quit|end|terminate|force\s+close)\s+"
        r"(?:the\s+)?(?:process\s+)?(.+?)(?:\s+(?:process|app))?[.!?]?$",
        t,
    )
    if m:
        name = m.group(1).strip().rstrip(".,!?")
        if name not in (
            "window",
            "panel",
            "hud",
            "overlay",
            "this",
            "app",
            "application",
            "it",
        ):
            return ("os_kill_process", {"name": name})

    # clipboard ─────────────────────────────────────────────────────────
    if re.search(
        r"(?:what(?:\'s|\s+is)\s+in\s+(?:my\s+)?clipboard|"
        r"read\s+clipboard|show\s+clipboard|get\s+clipboard|clipboard\s+content)",
        t,
    ):
        return ("os_get_clipboard", {})

    return None

"""Local phrase recognition shared by command resolution and the interfaces.

is_app_command protects established app phrases from shell-name correction.
_try_os_intent returns an action name and arguments for the capability registry;
matching a phrase neither executes that action nor grants permission for it.
Rule order is intentional because some natural-language patterns overlap.
"""

import re


def is_app_command(text):
    """Return whether bare input belongs to the existing application grammar.

    A prefix check keeps shell arguments such as ``git commit -m battery`` from
    being treated as OS requests. Explicit /exec routing is handled by callers.
    """
    stripped = text.strip()
    if stripped in {"ls", "tree"} or re.match(
        r"^(?:remind\s+me|read\s+file)\s+", stripped, re.I
    ):
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


def _try_os_intent(text: str):
    """Return (capability_name, argument_dict) for the first matching OS phrase.

    Unrecognized input returns None. Matching normalizes case and common aliases;
    validation and execution remain responsibilities of the capability gate.
    """
    t = text.lower().strip()

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

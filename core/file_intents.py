"""Recognize complete, empty-file requests without interpreting shell syntax.

The broad request guard also keeps unsupported file prose intact for the AI.
Recognition grants no permission: the capability gate validates and executes.
"""

import re


_POLITE_PREFIX = r"(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
_FILE_PREFIX = (
    _POLITE_PREFIX
    + r"(?:make|create)\s+(?:(?:a|an|the)\s+)?"
    r"(?:(?:new|empty)\s+){0,2}file\s+(?:(?:named|called|call)\s+)?"
)
_FILE_NAME = (
    r'''(?:"(?P<double>[^"\r\n]+)"|'(?P<single>[^'\r\n]+)'|'''
    r'''(?P<bare>[^\s"'<>|&;]+))'''
)
_LOCATION = (
    r"(?:in|on)\s+(?:(?:my|the)\s+)?"
    r"(?P<directory>desktop|documents|downloads|CreatedFolder|project|current(?:\s+working)?)"
    r"(?:\s+(?:folder|directory))?"
)
_CREATE_FILE = re.compile(
    _FILE_PREFIX + _FILE_NAME + rf"(?:\s+{_LOCATION})?"
    r"(?:\s+for\s+me)?(?:\s+please)?[.!?]?",
    re.I,
)
_FILE_REQUEST = re.compile(
    _POLITE_PREFIX + r"(?:make|create)\s+"
    r"(?:(?:(?:a|an|the|new|empty)\s+)(?:[\w-]+\s+){0,4})?"
    r"(?:files?|folders?|director(?:y|ies)|documents?|scripts?)(?=\s|$)",
    re.I,
)
_WRITE_REQUEST = re.compile(
    _POLITE_PREFIX + r"write\s+(?:(?:a|an|the|some|new)\s+)?"
    r"(?:(?:python|text|javascript|typescript)\s+)?"
    r"(?:code|script|file|program|text)(?=\s|$)", re.I,
)
_EDIT_REQUEST = re.compile(
    _POLITE_PREFIX + r"(?:edit|modify|change|update|rewrite)\s+"
    r"(?:(?:the|my)\s+)?(?:file\s+(?:(?:named|called)\s+)?)?"
    r'''(?:"[^"\r\n]+\.[a-z0-9]{1,16}"|'[^'\r\n]+\.[a-z0-9]{1,16}'|'''
    r'''[^\s"'<>|&;]+\.[a-z0-9]{1,16})'''
    r"\s+(?:so|that|to|with|by|and|using|such\s+that)\s+\S",
    re.I,
)


def try_file_intent(text: str):
    """Return one empty-file action only when the whole request is supported.

    Content instructions and additional actions must be handled as a whole by
    the AI; they must not silently become an empty file or a later OS action.
    Quoted file names retain their case and spaces without shell expansion.
    """
    request = text.strip()
    match = _CREATE_FILE.fullmatch(request)
    if match is None:
        return None
    name = match.group("double") or match.group("single") or match.group("bare")
    # Sentence punctuation after a bare filename is prose. Quoted names and
    # names followed by a destination keep their exact literal spelling.
    if match.group("bare") and match.end("bare") == len(request):
        name = name.rstrip(".!?")
        if not name:
            return None
    directory = (match.group("directory") or "project").lower()
    if directory.startswith("current") or directory == "createdfolder":
        directory = "project"
    return "create_empty_file", {"name": name, "directory": directory}


def is_file_request(text: str) -> bool:
    """Protect file prose, including edits, without interpreting its code.

    Edits require a filename plus a prose continuation so literal commands such
    as ``edit notes.txt`` or ``update script.py --check`` remain shell inputs.
    """
    return (_FILE_REQUEST.match(text.strip()) is not None
            or _WRITE_REQUEST.match(text.strip()) is not None
            or _EDIT_REQUEST.match(text.strip()) is not None)

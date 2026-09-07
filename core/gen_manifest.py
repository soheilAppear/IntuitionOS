"""Regenerate config/planner_schema.json from the live capability manifest.

    python -m core.gen_manifest

The checked-in schema used to be hand-maintained, and had drifted badly: it
advertised seven tools while nothing in the codebase parsed or dispatched a tool
call at all. Generating it means the advertised surface and the surface the gate
will actually permit cannot disagree.

This lives in its own module rather than under `if __name__ == "__main__"` in
core/capabilities.py because `python -m core.capabilities` would import that file
twice — once as __main__ and once as core.capabilities — leaving the __main__
copy with an empty registry and writing an empty schema.
"""

import json

from .actions import register_os_capabilities  # importing this registers the base manifest
from .capabilities import capabilities

DEFAULT_PATH = "config/planner_schema.json"


def generate(path: str = DEFAULT_PATH) -> str:
    register_os_capabilities()
    payload = {
        "_generated_by": "python -m core.gen_manifest",
        "_note": "Generated from core.capabilities. Do not edit by hand.",
        "tools": capabilities.manifest(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


if __name__ == "__main__":  # pragma: no cover
    print("wrote", generate())

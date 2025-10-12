# Simple logger that appends to a text file

import os, time

def make_logger(path:str):
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def log(msg:str):
        # Prepend time in ISO format
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    return log

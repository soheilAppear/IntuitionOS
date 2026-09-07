# Simple logger that appends to a text file

import os, time


def make_logger(path: str):
    """Append-to-file logger that cannot take down its caller.

    The directory is created lazily rather than at construction, and every write
    failure is swallowed. Both matter: the default logger is built at import time
    against a relative path, so anything that changes the working directory — an
    eval run, a test, a service started from elsewhere — leaves it pointing at a
    directory that does not exist. It used to raise from inside the gate's denial
    path, which meant a refused action crashed instead of being refused. A missing
    log line is a smaller problem than that in every case.
    """
    def log(msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {msg}\n")
        except Exception:
            pass

    return log

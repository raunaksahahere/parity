"""Record a command's terminal output as an asciicast v2 file (no external tools needed).

Usage: uv run python scripts/record_cast.py docs/demo.cast -- bash docs/demo.sh
"""

import json
import os
import pty
import select
import sys
import time


def main() -> None:
    out, cmd = sys.argv[1], sys.argv[sys.argv.index("--") + 1 :]
    header = {"version": 2, "width": 110, "height": 32, "timestamp": int(time.time()),
              "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"}}  # fmt: skip
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["COLUMNS"], os.environ["LINES"] = "110", "32"
        os.execvp(cmd[0], cmd)
    start = time.monotonic()
    events = []
    while True:
        r, _, _ = select.select([fd], [], [], 0.5)
        if fd in r:
            try:
                data = os.read(fd, 4096)
            except OSError:
                break
            if not data:
                break
            events.append([round(time.monotonic() - start, 3), "o", data.decode(errors="replace")])
            sys.stdout.write(data.decode(errors="replace"))
            sys.stdout.flush()
    os.waitpid(pid, 0)
    with open(out, "w") as f:
        f.write(json.dumps(header) + "\n")
        for e in events:
            f.write(json.dumps(e) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hermes on_session_start hook: catch-up consolidation when Desktop opens.

Register in ~/.hermes/config.yaml under hooks.on_session_start (setup.sh does
this). Whenever a chat session starts, poke the memory service to consolidate
any sessions still pending (missed by on_session_end — crash, force-quit, or
an LLM rate-limit at the time). The service debounces, so opening several
chats in a row costs at most one sweep per 10 minutes. Best-effort — a down
stack never breaks Hermes.
"""

import os
import sys
import urllib.request

URL = os.environ.get(
    "LONGBRAIN_MEMORY_URL", os.environ.get("HERMES_MEMORY_URL", "http://localhost:8800")
) + "/memory/consolidate-pending"


def main():
    sys.stdin.read()  # drain payload; content not needed
    request = urllib.request.Request(URL, data=b"{}",
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    try:
        urllib.request.urlopen(request, timeout=5)
    except Exception:
        pass  # best-effort


if __name__ == "__main__":
    main()
    print("{}")

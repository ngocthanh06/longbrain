import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # dotted `hooks.xxx` imports (e.g. hooks.admission_gate);
# `python -m pytest` from the repo root adds this automatically via cwd, but
# CI/containers invoke pytest with a different cwd (working-directory:
# llamaindex-service), where it's otherwise missing — masked locally by luck.
sys.path.insert(0, str(REPO_ROOT / "llamaindex-service"))  # the `app` package
sys.path.insert(0, str(REPO_ROOT / "hooks"))  # hook scripts (stdlib-only)
sys.path.insert(0, str(REPO_ROOT / "hooks" / "claude"))  # Claude Code adapter hooks
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # host-side scripts (stdlib-only)


class FakeEmbed:
    """Deterministic 2-dim embeddings: exact vectors per text, so tests can
    dial similarity precisely. Unknown texts fall back to [1, 0]."""

    def __init__(self, table: dict[str, list[float]] | None = None):
        self.table = table or {}

    def get_text_embedding(self, text: str) -> list[float]:
        return self.table.get(text, [1.0, 0.0])


class FakeCompletion:
    def __init__(self, text):
        self.text = text


class FakeLLM:
    """A canned-reply LLM stub — `reply` can be a fixed string or a callable
    (prompt) -> str for tests that need different answers per call."""

    def __init__(self, reply="yes"):
        self.reply = reply
        self.calls: list[str] = []

    def complete(self, prompt):
        self.calls.append(prompt)
        reply = self.reply(prompt) if callable(self.reply) else self.reply
        return FakeCompletion(reply)

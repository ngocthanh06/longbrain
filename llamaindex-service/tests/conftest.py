import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "llamaindex-service"))  # the `app` package
sys.path.insert(0, str(REPO_ROOT / "hooks"))  # hook scripts (stdlib-only)


class FakeEmbed:
    """Deterministic 2-dim embeddings: exact vectors per text, so tests can
    dial similarity precisely. Unknown texts fall back to [1, 0]."""

    def __init__(self, table: dict[str, list[float]] | None = None):
        self.table = table or {}

    def get_text_embedding(self, text: str) -> list[float]:
        return self.table.get(text, [1.0, 0.0])

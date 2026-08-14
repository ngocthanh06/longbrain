"""Unit tests for mcp_server.py request-model guards."""

import pydantic
import pytest

from app import config
from app.mcp_server import Fact


def test_fact_rejects_oversized_text():
    with pytest.raises(pydantic.ValidationError):
        Fact(text="x" * (config.MAX_FACT_TEXT_CHARS + 1))


def test_fact_rejects_empty_text():
    with pytest.raises(pydantic.ValidationError):
        Fact(text="")

"""Unit tests for the in-process write/destructive-action rate limiter
(app/ratelimit.py) — bounds a single runaway/looping agent call, see
config.py's write-path-guards comment for why this isn't per-user quota."""

import time

import pytest

from app import config, ratelimit


def test_allow_permits_calls_under_the_limit(monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_MAX_CALLS", 3)
    monkeypatch.setattr(config, "RATE_LIMIT_WINDOW_SECONDS", 10.0)
    ratelimit._calls.pop("test_under_limit", None)

    assert ratelimit.allow("test_under_limit")
    assert ratelimit.allow("test_under_limit")
    assert ratelimit.allow("test_under_limit")


def test_allow_refuses_once_the_limit_is_exceeded(monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_MAX_CALLS", 2)
    monkeypatch.setattr(config, "RATE_LIMIT_WINDOW_SECONDS", 10.0)
    ratelimit._calls.pop("test_over_limit", None)

    assert ratelimit.allow("test_over_limit")
    assert ratelimit.allow("test_over_limit")
    assert not ratelimit.allow("test_over_limit")


def test_allow_recovers_once_the_window_elapses(monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_MAX_CALLS", 1)
    monkeypatch.setattr(config, "RATE_LIMIT_WINDOW_SECONDS", 0.05)
    ratelimit._calls.pop("test_window_recovery", None)

    assert ratelimit.allow("test_window_recovery")
    assert not ratelimit.allow("test_window_recovery")
    time.sleep(0.06)
    assert ratelimit.allow("test_window_recovery")


def test_allow_tracks_actions_independently(monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_MAX_CALLS", 1)
    monkeypatch.setattr(config, "RATE_LIMIT_WINDOW_SECONDS", 10.0)
    ratelimit._calls.pop("test_action_a", None)
    ratelimit._calls.pop("test_action_b", None)

    assert ratelimit.allow("test_action_a")
    assert not ratelimit.allow("test_action_a")
    assert ratelimit.allow("test_action_b")


# ---------------------------------------------------------------------------
# Per-action tiers (Phase 3) — tiered actions must use their own config
# attribute, independent of the flat RATE_LIMIT_* pair used by everything
# else (recall/search/list, plus any future untiered action).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "action, attr",
    [
        ("memory_append", "RATE_LIMIT_APPEND_PER_MIN"),
        ("save_memories", "RATE_LIMIT_SAVE_PER_MIN"),
        ("add_to_knowledge_base", "RATE_LIMIT_INGEST_PER_MIN"),
        ("ingest", "RATE_LIMIT_INGEST_PER_MIN"),
        ("consolidate_session", "RATE_LIMIT_CONSOLIDATE_PER_MIN"),
        ("documents_delete", "RATE_LIMIT_DELETE_PER_MIN"),
        ("cleanup_garbage", "RATE_LIMIT_DELETE_PER_MIN"),
        ("forget_memory", "RATE_LIMIT_DELETE_PER_MIN"),
        ("forget_session", "RATE_LIMIT_DELETE_PER_MIN"),
        ("forget_everything", "RATE_LIMIT_FULL_RESET_PER_MIN"),
    ],
)
def test_tiered_action_uses_its_own_per_minute_budget(monkeypatch, action, attr):
    # A permissive flat fallback would let this test pass even if the tier
    # were wired wrong, so pin it far below the tier being tested.
    monkeypatch.setattr(config, "RATE_LIMIT_MAX_CALLS", 1000)
    monkeypatch.setattr(config, "RATE_LIMIT_WINDOW_SECONDS", 10.0)
    monkeypatch.setattr(config, attr, 2)
    ratelimit._calls.pop(action, None)

    assert ratelimit.allow(action)
    assert ratelimit.allow(action)
    assert not ratelimit.allow(action)


def test_tiered_action_window_is_sixty_seconds_not_the_flat_window():
    window, _max_calls = ratelimit._limit_for("save_memories")
    assert window == 60.0


def test_untiered_action_falls_back_to_the_flat_pair(monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_MAX_CALLS", 7)
    monkeypatch.setattr(config, "RATE_LIMIT_WINDOW_SECONDS", 42.0)
    assert ratelimit._limit_for("some_future_untiered_action") == (42.0, 7)

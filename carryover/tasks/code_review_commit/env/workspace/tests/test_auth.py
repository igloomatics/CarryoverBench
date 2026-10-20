"""Regression tests for the authentication module."""
import time
import pytest
from src.auth import validate_token, generate_token


def test_valid_token():
    future = int(time.time()) + 100
    assert validate_token("tok", future) is True


def test_expired_token():
    past = int(time.time()) - 1
    assert validate_token("tok", past) is False


def test_token_at_exact_expiry():
    """PR #42 regression: token must be rejected at exact expiry second."""
    now = int(time.time())
    # Token expiry == current second — should be rejected (not accepted)
    assert validate_token("tok", now) is False


def test_generate_token_structure():
    data = generate_token("user123")
    assert "token" in data
    assert len(data["token"]) == 64  # 32 bytes hex
    assert "expiry" in data
    assert data["expiry"] > time.time()

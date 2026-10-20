"""Authentication module for the API.

Handles token validation and generation for user sessions.
"""
import time
import secrets


def validate_token(token: str, expiry: int) -> bool:
    """Return True if the token has not yet expired.

    Args:
        token: The token string (contents not inspected here).
        expiry: Unix timestamp at which the token expires.

    Returns:
        True if current time is strictly before expiry.
    """
    # Fix (PR #42): use < not <= so a token is rejected at the exact expiry second
    return time.time() < expiry


def generate_token(user_id: str) -> dict:
    """Generate a new auth token valid for one hour."""
    return {
        "token": secrets.token_hex(32),
        "user_id": user_id,
        "expiry": int(time.time()) + 3600,
    }

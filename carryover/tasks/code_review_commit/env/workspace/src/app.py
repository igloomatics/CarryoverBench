"""Main application module."""

from .auth import validate_token, generate_token


def get_user(user_id: str, token: str, expiry: int) -> dict:
    """Return user data if the token is valid."""
    if not validate_token(token, expiry):
        raise PermissionError("Invalid or expired token")
    return {"user_id": user_id, "status": "active"}


def create_session(user_id: str) -> dict:
    """Create a new authenticated session."""
    token_data = generate_token(user_id)
    return {
        "session_id": token_data["token"][:16],
        "user_id": user_id,
        "expires_at": token_data["expiry"],
    }

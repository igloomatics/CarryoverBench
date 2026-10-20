from __future__ import annotations

from carryover.verifier.base import TaskEvaluator

_REGISTRY: dict[str, type[TaskEvaluator]] = {}


def register(cls: type[TaskEvaluator]) -> type[TaskEvaluator]:
    """Class decorator: register an evaluator by its task_id."""
    _REGISTRY[cls.task_id] = cls
    return cls


def load_evaluator(task_id: str) -> TaskEvaluator | None:
    """Return an evaluator instance for task_id, or None if not implemented.

    For multi-attack tasks the task_id is ``{base}__{mode}__{payload_id}``.
    We strip the suffix so all variants share the same utility evaluator.
    """
    base_id = task_id.split("__")[0]
    cls = _REGISTRY.get(base_id)
    return cls() if cls else None


# Import all concrete evaluators so they self-register via @register.
# One module per active benchmark task (task_id in carryover/tasks/<id>/).
from carryover.verifier import (  # noqa: E402, F401
    banking_transfer,
    code_review_commit,
    e_shopping,
    email_management,
    finance_tracking,
    food_delivery,
    health_management,
    literature_review,
    price_tracking,
    resume_submission,
    social_media,
    travel_booking,
)

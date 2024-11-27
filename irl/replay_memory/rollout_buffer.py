"""Base rollout buffer."""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class RolloutBuffer(list[T]):
    """Base Rollout Buffer."""

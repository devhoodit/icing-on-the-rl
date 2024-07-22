"""Base sampling policy."""

from __future__ import annotations

from abc import ABC, abstractmethod


class SamplingPolicy(ABC):
    """Sampling policy ABC."""

    @abstractmethod
    def is_random(self: SamplingPolicy, step: int) -> bool:
        """Is random with sampling policy in current step."""

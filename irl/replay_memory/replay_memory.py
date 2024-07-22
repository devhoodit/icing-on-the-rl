"""Basic replay memory."""

from __future__ import annotations

import random
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class ReplayMemory(Generic[T]):
    """Base replay memory."""

    def __init__(
        self: ReplayMemory,
        capacity: int,
    ) -> None:
        """Create replay memory.

        Args:
        ----
            capacity (int): replay queue capacity

        """
        self.capacity = capacity
        self.queue: deque[T] = deque([], self.capacity)

    def push(self: ReplayMemory, transition: T) -> None:
        """Push transition into memory."""
        self.queue.append(transition)

    def sample(self: ReplayMemory, batch_size: int) -> list[T]:
        """Sample batch from queue."""
        return random.sample(self.queue, batch_size)

    def __len__(self: ReplayMemory) -> int:
        """Return memory size."""
        return len(self.queue)

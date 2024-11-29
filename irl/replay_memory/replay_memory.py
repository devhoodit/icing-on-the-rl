"""Basic replay memory."""

from __future__ import annotations

import random
from collections import deque
from typing import Generic, TypeVar, MutableSequence, Any

T = TypeVar("T")

class BaseReplayMemory(Generic[T]):
    def __init__(self, factory: type[MutableSequence], **kwargs):
        self.factory = factory
        self.kwargs = kwargs
        self.memory = self._create_memory(self.factory, self.kwargs)
        
    def get_memory(self) -> MutableSequence[T]:
        return self.memory

    def push(self, transition: T) -> None:
        "push transition"
        self.memory.append(transition)
    
    def sample(self, batch_size: int) -> list[T]:
        raise NotImplementedError()
    
    def _create_memory(self, factory: type[MutableSequence], kwargs: dict[str, Any]) -> MutableSequence:
        if not kwargs:
            return factory()
        return factory(**kwargs)
    
    def clear(self) -> None:
        self.memory = self._create_memory(self.factory, self.kwargs)
    
    def __len__(self) -> int:
        """memory length"""
        return len(self.memory)


class ReplayMemory(BaseReplayMemory[T]):
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
        super().__init__(deque, {"maxlen": capacity})
        self.capacity = capacity

    def sample(self: ReplayMemory, batch_size: int) -> list[T]:
        """Sample batch from queue."""
        return random.sample(self.memory, batch_size)

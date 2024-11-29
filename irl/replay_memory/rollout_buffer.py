"""Base rollout buffer."""

from __future__ import annotations

from typing import TypeVar
from irl.logger import get_module_logger
from .replay_memory import BaseReplayMemory

T = TypeVar("T")

class RolloutBuffer(BaseReplayMemory[T]):
    """Base Rollout Buffer."""
    def __init__(self, is_log=True):
        super().__init__(list)
        self.is_log = True
    
    def sample(self, batch_size) -> list[T]:
        """rollout buffer sample always return empty list"""
        if self.is_log:
            get_module_logger().warning("rollout buffer sample always return empty list, to turn off this message set RolloutBuffer(is_log=False) or set module log level greater than warning")
        return []
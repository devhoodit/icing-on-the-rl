"""Hindsight Experience Replay (HER)."""

from .replay_memory import BaseReplayMemory
from typing import TypeVar

T = TypeVar("T")

class HER(BaseReplayMemory[T]):
    def __init__(self):
        super().__init__(list)
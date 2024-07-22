"""RL methods.

DQL.
"""

from irl.agent import agent
from irl.methods import eval, hook, qlearning

__all__ = ["qlearning", "eval", "hook", "agent"]

"""Egreedy policies."""

import math
import random

from .policy import SamplingPolicy


class ExponentialEgreedyPolicy(SamplingPolicy):
    """Exponential egreedy sampling policy."""

    def __init__(self, initial: float, decay: int, minimum: float) -> None:
        """Init Exponential policy.

        Args:
        ----
            initial (float): start value
            decay (int): decay during step range
            minimum (float): minimum value

        """
        super().__init__()
        self.initial = initial
        self.decay = decay
        self.minimum = minimum

    def is_random(self, step: int) -> bool:
        """Return is random in step."""
        eps_threshold = self.minimum + (self.initial - self.minimum) * math.exp(
            -1.0 * step / self.decay,
        )
        sample = random.random()
        return sample < eps_threshold

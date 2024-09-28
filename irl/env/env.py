from __future__ import annotations

from typing import Any, Sequence, SupportsFloat, SupportsInt, TypeVar

import numpy as np

ObsType = np.ndarray
ActionType = np.ndarray | int | float


class Env:
    """Iciing on the RL env wrapper."""

    def __init__(
        self,
    ) -> None:
        """Initialize env."""

    def reset(self) -> ObsType:
        """Env reset.

        Returns
        -------
            ObsType: Env return observation space.

        """
        msg = "reset method not implemented"
        raise NotImplementedError(msg)

    def step(
        self,
        action: ActionType,  # noqa: ARG002
    ) -> tuple[
        ObsType,
        float,
        bool,
        bool,
    ]:
        """Env step.

        Args:
        ----
            action (ActionType): env action and return observation space and reward

        Returns:
        -------
            tuple[ ObsType, float, bool, bool, ]: return observation space, reward, terminate, truncated

        """
        msg = "step not implemented"
        raise NotImplementedError(msg)

    def select_random_action(self) -> ActionType:
        """Select random action.

        Returns
        -------
            ActionType: return random action

        """
        msg = "select random action not implemented"
        raise NotImplementedError(msg)


class Discrete:
    """Discrete space."""

    def __init__(
        self,
        elems: Sequence[Any],
    ) -> None:
        """Discrete space.

        Args:
        ----
            elems (Sequence[Any]): Sequence of elems.

        """
        self.elems = np.array(elems)
        self.dtype = self.elems.dtype


class RangeContinuous:
    """Range continous space."""

    def __init__(
        self,
        start: SupportsFloat,
        end: SupportsFloat,
        dtype: type[np.float64 | np.float32 | np.float16] = np.float64,
    ) -> None:
        """Continuous space.

        Args:
        ----
            start (SupportsFloat): start point of space
            end (SupportsFloat): end point of space
            dtype (type[np.float64  |  np.float32  |  np.float16], optional): precision of space. Defaults to np.float64.

        """
        self.start = dtype(start)
        self.end = dtype(end)
        self.dtype = dtype


class Continous:
    """No range continous."""

    def __init__(self) -> None:
        """Initialize."""

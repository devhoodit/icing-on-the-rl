"""hook for training."""

from irl.env.env import ActionType, ObsType


class BaseHook:
    """Base hook."""

    def before_episode(self, epi_n: int) -> None:
        """Call hook before episode.

        Args:
        ----
            epi_n (int): episode step

        """

    def after_episode(self, epi_n: int, score: float) -> None:
        """Call hook after episode.

        Args:
        ----
            epi_n (int): episode step
            score (float): total score of episode

        """

    def before_step(self, step_n: int, state: ObsType, action: ActionType) -> None:
        """Call hook before step.

        Args:
        ----
            step_n (int): step in total env
            state (ObsType): state before select action
            action (ActionType): selected action

        """

    def after_step(
        self,
        step_n: int,
        obs: ObsType,
        reward: float,
    ) -> None:
        """Call hook after step.

        Args:
        ----
            step_n (int): step in total env
            obs (ObsType): state after action
            reward (float): reward after action

        """

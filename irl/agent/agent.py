"""Base agent."""

from irl.env.env import ActionType, ObsType


class Agent:
    """RL agent base class."""

    def select_action(self, state: ObsType) -> ActionType:  # noqa: ARG002
        """Select best action from current state.

        Returns
        -------
            ActionType: return best action with current agent policy

        """
        msg = "select action is not implemented"
        raise NotImplementedError(msg)

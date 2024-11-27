"""REINFORCE agent and method."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.distributions import Categorical, Distribution, Normal

from irl.agent.agent import Agent
from irl.env.env import ObsType
from irl.hook import BaseHook

if TYPE_CHECKING:
    from irl.env.env import ActionType, Env, ObsType

distribution_callback = Callable[[Tensor], Distribution]


def __normal_dist_generator(x: Tensor) -> Distribution:
    return Normal(x[0].item(), x[1].item())


class REINFORCEAgent(Agent):
    """REINFORCE agent base class."""

    def __init__(
        self,
        net: nn.Module,
        device: str,
    ) -> None:
        """Initialize ANN."""
        self.net = net
        self.device = device

    def sample_action_and_prob(self, state: ObsType) -> tuple[ActionType, float]:  # noqa: ARG002
        """Sample action from state and return log probability from action.

        Args:
        ----
            state (ObsType): observation state

        Raises:
        ------
            NotImplementedError: abstract method

        Returns:
        -------
            tuple[ActionType, float]: action, log probability

        """
        msg = "select_action not implemented, recommend to use REINFORCEAgentDiscrete or REINFORCEAgentContinous"
        raise NotImplementedError(msg)

    def select_action(self, state: ObsType) -> ActionType:  # noqa: ARG002, D102
        msg = "select_action not implemented, recommend to use REINFORCEAgentDiscrete or REINFORCEAgentContinous"
        raise NotImplementedError(msg)


class REINFORCEAgentDiscrete(REINFORCEAgent):
    """REINFORCE discrete action space agent."""

    def __init__(self, net: nn.Module, device: str) -> None:
        """Initialize agent.

        Args:
        ----
            net (nn.Module): ANN
            device (str): device type

        """
        super().__init__(net, device)

    def sample_action_and_prob(self, state: ObsType) -> tuple[ActionType, float]:
        """Sample action and log probability.

        Args:
        ----
            state (ObsType): observation state

        Returns:
        -------
            tuple[ActionType, float]: action and log probability

        """
        x = torch.from_numpy(state)
        output: Tensor = self.net(x)
        prob = F.softmax(output, dim=0)
        prob_distribution = Categorical(prob)
        action = prob_distribution.sample()
        return action.item(), prob_distribution.log_prob(action).item()

    def sample_action(self, state: ObsType) -> ActionType:
        """Sample action.

        Args:
        ----
            state (ObsType): observation state

        Returns:
        -------
            ActionType: action

        """
        action, _ = self.sample_action_and_prob(state)
        return action


class REINFORCEAgentContinous(REINFORCEAgent):
    """REINFORCE continous agent."""

    def __init__(self, net: nn.Module, device: str, distribution_generator: distribution_callback | None = None) -> None:
        """Initialize.

        Args:
        ----
            net (nn.Module): ANN
            device (str): device type
            distribution_generator (distribution_callback | None, optional): . Defaults to None.

        """
        super().__init__(net, device)
        if distribution_generator is None:
            distribution_generator = __normal_dist_generator
        self.distribution_generator = distribution_generator

    def sample_action_and_prob(self, state: ObsType) -> tuple[ActionType, float]:
        """Sample action and log probability.

        Args:
        ----
            state (ObsType): observation state

        Returns:
        -------
            tuple[ActionType, float]: action and log probability

        """
        x = torch.from_numpy(state)
        output: Tensor = self.net(x)
        prob_distribution = self.distribution_generator(output)
        action = prob_distribution.sample()
        return action.item(), prob_distribution.log_prob(action).item()

    def sample_action(self, state: ObsType) -> ActionType:
        """Sample action.

        Args:
        ----
            state (ObsType): observation state

        Returns:
        -------
            ActionType: action

        """
        action, _ = self.sample_action_and_prob(state)
        return action


class REINFORCEOptimizer:
    """REINFORCE optimizer for optimize REINFORCE agent."""

    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str,
        gamma: float,
    ) -> None:
        """Initialize REINFORCE optimizer.

        Args:
        ----
            net (nn.Module): network
            criterion (nn.Module): criterion
            optimizer (optim.Optimizer):optimizer
            device (str): device
            gamma (float): gamma

        """
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma

        self.reward_history = []
        self.prob_history = []

    def reset_history(self) -> None:
        """Reset reward and prob history."""
        self.reward_history = []
        self.prob_history = []

    def save_reward_and_log_prob(self, reward: float, prob: float) -> None:
        """Save reward and prob history for optimize after episode."""
        self.reward_history.append(reward)
        self.prob_history.append(prob)

    def optimize(self) -> None:
        """Optimize network with saved reward and prob history."""
        self.optimizer.zero_grad()
        dr = 0
        for r, prob in zip(self.reward_history[::-1], self.prob_history[::-1]):
            dr = r + self.gamma * dr
            loss = -dr * prob
            loss.backward()
        self.optimizer.step()


class REINFORCE:
    """REINFORCE train ground."""

    def __init__(self, reinforce_agent: REINFORCEAgent, env: Env) -> None:
        """Initialize env, agent."""
        self.env = env
        self.agent = reinforce_agent

    def train(self, *, episode: int, reinforce_optimizer: REINFORCEOptimizer, hook: BaseHook | None = None) -> None:
        """Train REINFORCE agent.

        Args:
        ----
            episode (int): train episode
            reinforce_optimizer (REINFORCEOptimizer): REINFORCE optimizer
            hook (BaseHook | None, optional): train hook. Defaults to None.

        """
        if hook is None:
            hook = BaseHook()

        total_step = 0
        for epi_n in range(episode):
            score = 0.0
            hook.before_episode(epi_n)
            state = self.env.reset()
            while True:
                action, prob = self.agent.sample_action_and_prob(state)
                hook.before_step(total_step, state, action)
                state, reward, terminated, truncated = self.env.step(action)
                total_step += 1
                hook.after_step(total_step, state, reward)
                score += reward
                reinforce_optimizer.save_reward_and_log_prob(reward, prob)

                if terminated or truncated:
                    break

            reinforce_optimizer.optimize()
            reinforce_optimizer.reset_history()
            hook.after_episode(epi_n, score)

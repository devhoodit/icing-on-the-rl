"""PPO agent and method."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor, optim
from torch.distributions import Categorical, Distribution

from irl.agent.agent import Agent
from irl.hook import BaseHook
from irl.replay_memory.rollout_buffer import RolloutBuffer

if TYPE_CHECKING:
    from irl.env.env import ActionType, Env, ObsType


class PPOBufferCollection(NamedTuple):
    """PPO rollout buffer elements."""

    action: Tensor
    state: Tensor
    log_prob: Tensor
    reward: float
    state_value: Tensor
    is_terminal: bool

    @staticmethod
    def list_to_batch(
        buffers: list[PPOBufferCollection],
    ) -> PPOBufferCollectionBatch:
        """Wrap PPOBufferCollection list to PPOBufferCollectionBatch.

        Args:
        ----
            buffers (list[PPOBufferCollection]): buffers list

        Returns:
        -------
            PPOBufferCollectionBatch: _description_

        """
        action_batch, state_batch, log_prob_batch, reward_batch, state_value_batch, is_terminal_batch = zip(*buffers)
        return PPOBufferCollectionBatch(
            action_batch,
            state_batch,
            log_prob_batch,
            reward_batch,
            state_value_batch,
            is_terminal_batch,
        )


class PPOBufferCollectionBatch(NamedTuple):
    """PPOBufferCollectionBatch for learning rollout."""

    actions: tuple[Tensor]
    states: tuple[Tensor]
    log_probs: tuple[Tensor]
    rewards: tuple[float]
    state_values: tuple[Tensor]
    is_terminals: tuple[bool]


class PPOAgent(Agent):
    """PPO Agent."""

    def __init__(self, actor_net: nn.Module, critic_net: nn.Module) -> None:
        """Initialize agent.

        Args:
        ----
            actor_net (nn.Module): actor network
            critic_net (nn.Module): critic network

        """
        self.actor_net = actor_net
        self.critic_net = critic_net

    def sample_action_and_prob_and_state_value(self, state: ObsType) -> tuple[Tensor, Tensor, Tensor]:
        """Sample action with probs and state value from state.

        Args:
        ----
            state (ObsType): state

        Raises:
        ------
            NotImplementedError: need to implemented based on agent type

        Returns:
        -------
            action (Tensor): action
            probs (Tensor): probs
            state_value (Tensor): state value

        """
        raise NotImplementedError

    def select_action(self, state: ObsType) -> ActionType:
        """Select action from state.

        Args:
        ----
            state (ObsType): state

        Raises:
        ------
            NotImplementedError: need to implemented based on agent type

        Returns:
        -------
            action (ActionType): action

        """
        raise NotImplementedError

    def evaluate(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate log probs, state values and entropies from state and action.

        Args:
        ----
            state (Tensor): state
            action (Tensor): action

        Raises:
        ------
            NotImplementedError: need to implemented based on agent type

        Returns:
        -------
            action_log_prob (Tensor): log prob
            state_value (Tensor): state value
            entropy (Tensor): entropy

        """
        raise NotImplementedError

    def load_from_agent(self, agent: PPOAgent) -> None:
        """Load agent from another agent."""
        self.actor_net.load_state_dict(agent.actor_net.state_dict())
        self.critic_net.load_state_dict(agent.critic_net.state_dict())

    def build_net_optim_params(self, actor_lr: float, critic_lr: float) -> list:
        return [
            {"params": self.actor_net.parameters(), "lr": actor_lr},
            {"params": self.critic_net.parameters(), "lr": critic_lr},
        ]


class PPODiscreteAgent(PPOAgent):
    def __init__(self, actor_net: nn.Module, critic_net: nn.Module, device: str) -> tuple[Tensor, Tensor, Tensor]:  # type: ignore
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.device = device

    def sample_action_dist(self, state: Tensor) -> Distribution:
        action_probs = self.actor_net(state)
        return Categorical(action_probs)

    def sample_action_and_prob_and_state_value(self, state: ObsType) -> tuple[Tensor, Tensor, Tensor]:
        t_state = torch.from_numpy(state).to(self.device)
        dist = self.sample_action_dist(t_state)
        action = dist.sample()
        action_log_prob: Tensor = dist.log_prob(action)
        state_value: Tensor = self.critic_net(t_state)

        return action.detach(), action_log_prob.detach(), state_value.detach()

    def select_action(self, state: ObsType) -> ActionType:
        with torch.no_grad():
            action, action_log_prob, state_value = self.sample_action_and_prob_and_state_value(state)
            return action.item()

    def evaluate(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dist = self.sample_action_dist(state)
        action_log_probs = dist.log_prob(action)
        state_values = self.critic_net(state)
        dist_entropy = dist.entropy()
        return action_log_probs, state_values, dist_entropy


class PPOOptimizer:
    def __init__(
        self,
        agent: PPOAgent,
        criterion: nn.Module,
        optim: optim.Optimizer,
        gamma: float,
        epoch: int,
        device: str,
        eps_clip: float,
    ) -> None:
        self.agent = agent
        self.old_agent = deepcopy(agent)
        self.old_agent.actor_net.to(device)
        self.old_agent.critic_net.to(device)
        self.criterion = criterion
        self.optim = optim
        self.gamma = gamma
        self.epoch = epoch
        self.device = device
        self.eps_clip = eps_clip

    def update(self, buffer: RolloutBuffer[PPOBufferCollection]) -> None:
        rewards = []
        discounted_reward = 0

        batch = PPOBufferCollection.list_to_batch(buffer)

        for reward, is_terminal in zip(reversed(batch.rewards), reversed(batch.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(batch.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(batch.actions, dim=0)).detach().to(self.device)
        old_lob_probs = torch.squeeze(torch.stack(batch.log_probs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(batch.state_values, dim=0)).detach().to(self.device)

        advantages = rewards.detach() - old_state_values.detach()

        for _ in range(self.epoch):
            log_probs, state_values, dist_entropy = self.agent.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(log_probs - old_lob_probs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.criterion(state_values, rewards) - 0.01 * dist_entropy
            self.optim.zero_grad()
            loss.mean().backward()
            self.optim.step()
        self.old_agent.load_from_agent(self.agent)
        buffer.clear()


class PPO:
    """PPO train ground."""

    def __init__(self, ppo_agent: PPOAgent, env: Env, device: str) -> None:
        """Initialize env, agent."""
        self.agent = ppo_agent
        self.env = env
        self.device = device

    def train(
        self,
        ppo_optimizer: PPOOptimizer,
        episode: int,
        update_iter: int,
        max_iter: int | None = None,
        hook: BaseHook | None = None,
    ) -> None:
        """Train PPO agent."""
        if hook is None:
            hook = BaseHook()

        buffer = RolloutBuffer[PPOBufferCollection]()
        total_step = 0
        train_done = False
        for epi_n in range(episode):
            score = 0.0
            hook.before_episode(epi_n)
            state = self.env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    action, action_log_prob, state_value = ppo_optimizer.old_agent.sample_action_and_prob_and_state_value(state)
                hook.before_step(total_step, state, action.detach().item())
                state, reward, terminate, truncated = self.env.step(action.item())
                score += reward
                total_step += 1
                hook.after_step(total_step, state, reward)
                done = terminate or truncated

                buffer.append(
                    PPOBufferCollection(
                        torch.from_numpy(state),
                        action,
                        action_log_prob,
                        reward,
                        state_value,
                        done,
                    ),
                )

                if total_step % update_iter == 0:
                    ppo_optimizer.update(buffer)
                    buffer.clear()

                if max_iter is not None and total_step >= max_iter:
                    train_done = True
                    break
            hook.after_episode(epi_n, score)
            if train_done:
                break

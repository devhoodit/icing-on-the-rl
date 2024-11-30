"""DQL agent and method."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, NamedTuple, SupportsFloat, SupportsInt

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn.utils.clip_grad import clip_grad_value_

from irl.agent import Agent
from irl.hook import BaseHook
from irl.methods.eval import eval_agent

if TYPE_CHECKING:
    from irl.env.env import ActionType, Env, ObsType
    from irl.replay_memory.replay_memory import ReplayMemory
    from irl.sampling.policy import SamplingPolicy


class Transition(NamedTuple):
    """Q learning transition."""

    state: np.ndarray
    action: SupportsInt
    reward: SupportsFloat
    next_state: np.ndarray | None

    @staticmethod
    def list_to_batch(
        transitions: list[Transition],
    ) -> TransitionBatch:
        """Wrap Transitions list to TransitionBatch.

        Args:
        ----
            transitions (list[Transition]): transitions list

        Returns:
        -------
            TransitionBatch: transition batch for learning

        """
        state_batch, action_batch, reward_batch, next_state_batch = zip(*transitions)
        return TransitionBatch(state_batch, action_batch, reward_batch, next_state_batch)


class TransitionBatch(NamedTuple):
    """TransitionBatch for learning in batch."""

    state: tuple[np.ndarray]
    action: tuple[SupportsFloat]
    reward: tuple[int]
    next_state: tuple[np.ndarray | None]


class DQLAgent(Agent):
    """DQL Agent."""

    def __init__(
        self,
        net: nn.Module,
        device: str,
    ) -> None:
        """Initialize agent.

        Args:
        ----
            net (nn.Module): network
            device (str): neural network device type

        """
        self.net = net
        self.device = device

    def select_action(self, state: ObsType) -> ActionType:
        """Select greedy action.

        warning: not affected by sampling policy, only take greedy so when train you must add policy condition.

        Args:
        ----
            state (Tensor): observation space

        """
        with torch.no_grad():
            output: Tensor = self.net(torch.from_numpy(state).unsqueeze(0).to(self.device))
            return output.argmax().to("cpu").numpy()


class DQNOptimizer:
    """DQN optimizer for seperating optimizing step from traning."""

    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str,
        gamma: float,
        target_net: nn.Module | None = None,
        tau: float = 1.0,
        clip_grad: float = 100.0,
    ) -> None:
        """Initialize learning agent configuration.

        Args:
        ----
            net (nn.Module): neural net
            criterion (nn.Module): ciriterion
            optimizer (optim.Optimizer): optimizer
            device (str): net device type
            gamma (float): reward gamma
            target_net (nn.Module | None, optional): policy net for Double DQN (DDQN). Defaults to None.
            tau (float, optional): target net update rate for DDQN. If policy net is None, never work. Defaults to 1.0.
            clip_grad (float, optional): clip grad value of parameters. Defaults to 100.0

        """
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma
        self.target_net = target_net
        self.tau = tau
        self.optimize_count = 0
        self.clip_grad = clip_grad
        if self.tau != 1.0 and self.target_net is None:
            warnings.warn("policy net is not None, but tau is 1.0. update net smoothly is not working", stacklevel=1)

    def initialize_optimize_count(self) -> None:
        """Initialize optimize_count to 0."""
        self.optimize_count = 0

    def optimize(
        self,
        transition_batch: TransitionBatch,
    ) -> None:
        """Optimize with transition batch.

        update optimize count += 1.

        Args:
        ----
            transition_batch (TransitionBatch): transition batch
            clip_grad (float, optional): clip gradiant range (-clip_grad, +clip_grad). see https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html
            update_target_net (bool, optional): update target network. If False, update not work. Defaults to True.

        """
        # sample from replay memory
        batch = transition_batch
        batch_size = len(transition_batch.state)
        arr_type = batch.state[0].dtype

        non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state), device=self.device, dtype=torch.bool)
        non_final_next_state = torch.stack([torch.from_numpy(s) for s in batch.next_state if s is not None]).to(self.device)

        state_batch = torch.from_numpy(np.array(batch.state, dtype=arr_type)).to(self.device)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(np.array(batch.reward, dtype=arr_type)).to(self.device)
        target_net = self.net if self.target_net is None else self.target_net
        # if target network has smooth update policy get policy network
        # get Q(s, a)
        state_action_values = self.net(state_batch).gather(1, action_batch)
        # get V(s')
        next_state_values = torch.zeros(
            batch_size,
            device=self.device,
            dtype=torch.float,
        )
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_state).max(1).values  # noqa: PD011

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = self.criterion(
            state_action_values,
            expected_state_action_values.unsqueeze(1),
        )
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_value_(self.net.parameters(), self.clip_grad)
        self.optimizer.step()

        # if smooth update network
        if self.target_net is not None:
            policy_net_state_dict = self.net.state_dict()
            target_net_state_dict = self.target_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
            self.target_net.load_state_dict(target_net_state_dict)


class DQL:
    """DQL train ground."""

    def __init__(self, dql_agent: DQLAgent, env: Env, device: str) -> None:
        """Initialize env, agent."""
        self.env = env
        self.device = device
        self.agent = dql_agent

    def train(
        self,
        *,
        episode: int,
        dqn_optimizer: DQNOptimizer,
        batch_size: int,
        sampling_policy: SamplingPolicy,
        replay_memory: ReplayMemory[Transition],
        hook: BaseHook | None = None,
    ) -> None:
        """Train DQL agent.

        Args:
        ----
            episode (int): train episode
            dqn_optimizer (DQNOptimizer): DQN agent optimizer for optimize batch
            batch_size (int): sample batch size from replay memory
            sampling_policy (SamplingPolicy): action sampling policy for exploration and exploitation
            replay_memory (ReplayMemory[Transition]): replay memory
            hook (BaseHook | None, optional): hook. Defaults to None.

        """
        if hook is None:
            hook = BaseHook()

        total_step = 0
        for epi_n in range(episode):
            score = 0.0
            hook.before_episode(epi_n)
            state = self.env.reset()
            done = False
            while not done:
                if sampling_policy.is_random(total_step):
                    selected_action = self.env.select_random_action()
                else:
                    selected_action = self.agent.select_action(state)

                hook.before_step(total_step, state, selected_action)
                next_state, reward, terminated, truncated = self.env.step(
                    selected_action,
                )
                total_step += 1
                hook.after_step(total_step, next_state, reward)
                score += reward
                # push for batch learning
                replay_memory.push(
                    Transition(
                        state,
                        selected_action,
                        reward,
                        next_state if not terminated else None,
                    ),
                )

                state = next_state

                if len(replay_memory) > batch_size:
                    sampled_batch = Transition.list_to_batch(replay_memory.sample(batch_size))
                    dqn_optimizer.optimize(sampled_batch)

                done = terminated or truncated
            hook.after_episode(epi_n, score)

    def eval(self, episode: int, hook: BaseHook | None = None) -> None:
        """Eval agent.

        Args:
        ----
            episode (int): _description_
            hook (BaseHook | None, optional): _description_. Defaults to None.

        """
        eval_agent(self.env, self.agent, episode, hook)

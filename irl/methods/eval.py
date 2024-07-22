"""eval agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from irl.methods.hook import BaseHook

if TYPE_CHECKING:
    from irl.agent.agent import Agent
    from irl.env.env import Env


def eval_agent(
    env: Env,
    agent: Agent,
    episode: int,
    hook: BaseHook | None = None,
) -> None:
    """Eval agent."""
    if hook is None:
        hook = BaseHook()

    step = 0
    for epi_n in range(episode):
        score = 0.0
        hook.before_episode(epi_n)
        state = env.reset()
        while True:
            selected_action = agent.select_action(state)
            hook.before_step(step, state, selected_action)
            state, reward, terminated, truncated = env.step(
                selected_action,
            )
            step += 1
            hook.after_step(step, state, reward)
            score += float(reward)
            if terminated or truncated:
                break
        hook.after_episode(epi_n, score)

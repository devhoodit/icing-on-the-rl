from irl.agent import Agent
from irl.env import Env
from irl.hook import BaseHook
from irl.replay_memory import BaseReplayMemory

from typing import TypeVar, Generic, Callable
from enum import Enum

M = TypeVar("M", BaseReplayMemory)

# episode, step, total_step, terminate, truncated -> is_stop_collect
collect_end_check_fn = Callable[[int, int, int, bool, bool], bool]

class CollectEndCheckFnCollection:
    @staticmethod
    def create_episode_per_check_fn(freq_epi: int) -> collect_end_check_fn:
        def collect_episodic_check_fn(epi: int, step: int, total_step: int, terminate: bool, truncated: bool) -> bool:
            return epi % freq_epi == 0
        
        return collect_episodic_check_fn
    
    @staticmethod
    def create_step_per_epi_check_fn(freq_step: int) -> collect_end_check_fn:
        def collect_step_per_epi_check_fn(epi: int, step: int, total_step: int, terminate: bool, truncated: bool) -> bool:
            return step % freq_step == 0
        
        return collect_step_per_epi_check_fn
    
    @staticmethod
    def create_step_check_fn(freq_step: int) -> collect_end_check_fn:
        def collection_step_check_fn(epi: int, step: int, total_step: int, terminate: bool, truncated: bool) -> bool:
            return total_step % freq_step == 0
        
        return collection_step_check_fn
    
    @staticmethod
    def create_episode_done_check_fn() -> collect_end_check_fn:
        def collect_episode_done_check_fn(epi: int, step: int, total_step: int, terminate: bool, truncated: bool) -> bool:
            return terminate or truncated

        return collect_episode_done_check_fn
    
        
    


class OnPolicyRolloutCollector(Generic[M]):
    def __init__(self, env: Env, agent: Agent, replay_memory: M, hook: BaseHook, collect_end_fn: collect_end_check_fn):
        self.env = env
        self.agent = agent
        self.replay_memory = replay_memory
        self.hook = hook
        self.collect_end_fn = collect_end_check_fn

        self.step = 0
        self.epi = 0
        self.total_step = 0

    def collect(self) -> M:
        
        return self.replay_memory


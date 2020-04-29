from collections import defaultdict
from dataclasses import dataclass

from .variable_wrapper import VariableWrapper


@dataclass
class EpisodeInfo:
    total_true_reward: float
    length: int


class VariableMonitor(VariableWrapper):
    """
    Use to automatically fill total_true_reward and episode_length of VariableStepResult
    """
    def __init__(self, env):
        super().__init__(env)
        self._episodes = defaultdict(EpisodeInfo)

    def reset(self):
        self._episodes.clear()
        return self.env.reset()

    def step(self, action):
        data = self.env.step(action)

        for i in range(len(data.obs)):
            agent_id = data.agent_id[i]
            ep_info = self._episodes[agent_id]
            if data.done[i]:
                del self._episodes[agent_id]
            else:
                ep_info.length += 1
            ep_info.total_true_reward += data.true_reward[i]
            data.total_true_reward[i] = ep_info.total_true_reward
            data.episode_length[i] = ep_info.length

        return data
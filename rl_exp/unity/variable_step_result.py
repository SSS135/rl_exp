from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from mlagents_envs.base_env import AgentId


@dataclass
class VariableStepResult:
    obs: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    max_step: np.ndarray
    agent_id: np.ndarray
    true_reward: np.ndarray
    total_true_reward: Optional[np.ndarray]
    episode_length: Optional[np.ndarray]
    _agent_id_to_index: Optional[Dict[AgentId, int]] = field(init=False, repr=False, default=None)

    @property
    def agent_id_to_index(self) -> Dict[AgentId, int]:
        if self._agent_id_to_index is None:
            self._agent_id_to_index = {}
            for a_idx, a_id in enumerate(self.agent_id):
                self._agent_id_to_index[a_id] = a_idx
        return self._agent_id_to_index

    def contains_agent(self, agent_id: AgentId) -> bool:
        return agent_id in self.agent_id_to_index
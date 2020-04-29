import uuid
from enum import Enum
from typing import Dict, List, Tuple, NamedTuple

from mlagents_envs.side_channel import SideChannel, IncomingMessage


class AuxRewardType(Enum):
    true_reward = 0
    living = 1
    kill = 3,
    death = 4,
    damage_received = 5,
    damage_dealt = 6
    # enemy_approaching = 7,


class AuxRewards(NamedTuple):
    agent_id: int
    step: int
    rewards: List[float]


class AuxRewardSideChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID('ffc44709-4e03-4a6e-a635-60dead4d8ff9'))
        self._rewards: Dict[int, Tuple[int, List[float]]] = dict()

    def on_message_received(self, msg: IncomingMessage) -> None:
        agent_id = msg.read_int32()
        step = msg.read_int32()
        new_rewards = msg.read_float32_list()
        if agent_id in self._rewards:
            last_step, last_rewards = self._rewards[agent_id]
            assert last_step == -1 or last_step == step
            for i in range(len(new_rewards)):
                new_rewards[i] += last_rewards
        self._rewards[agent_id] = (step, new_rewards)

    def collect_rewards(self) -> List[AuxRewards]:
        rewards = [AuxRewards(id, step, rewards) for id, (step, rewards) in self._rewards.items()]
        self._rewards.clear()
        return rewards

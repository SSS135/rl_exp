import struct
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Dict, List, Tuple, NamedTuple
import numpy as np

from mlagents_envs.base_env import AgentId
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage


@unique
class AuxRewardType(Enum):
    true_reward = 0
    living = 1
    kill = 3,
    death = 4,
    damage_received = 5,
    damage_dealt = 6
    # enemy_approaching = 7,


class AuxRewardSideChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID('ffc44709-4e03-4a6e-a635-60dead4d8ff9'))
        self._rewards: Dict[AgentId, np.ndarray] = dict()

    def on_message_received(self, msg: IncomingMessage) -> None:
        agent_id = msg.read_int32()
        reward_type = msg.read_int32()
        reward = msg.read_float32()
        if agent_id not in self._rewards:
            self._rewards[agent_id] = np.zeros(len(AuxRewardType), dtype=np.float32)
        self._rewards[agent_id][reward_type] += reward

    def collect_rewards(self, step_agents: np.ndarray) -> np.ndarray:
        # assert len(set(step_agents) - set(self._rewards.keys())) == 0, (set(self._rewards.keys()), set(step_agents))
        rewards = np.zeros((len(step_agents), len(AuxRewardType)), dtype=np.float32)
        for i, aid in enumerate(step_agents):
            if aid in self._rewards:
                rewards[i] = self._rewards[aid]
        self._rewards.clear()
        return rewards


def test_AuxRewardSideChannel():
    channel = AuxRewardSideChannel()

    def construct_msg(agent_id, reward_type, reward):
        m = OutgoingMessage()
        m.write_int32(agent_id)
        m.write_int32(reward_type)
        m.write_float32(reward)
        return IncomingMessage(m.buffer)

    channel.on_message_received(construct_msg(0, 1, -1))
    channel.on_message_received(construct_msg(7, 2, 2))
    channel.on_message_received(construct_msg(7, 2, 3))

    rewards = channel.collect_rewards(np.array([0, 7]))
    assert np.allclose(rewards, np.asarray([[0, -1, 0, 0, 0, 0], [0, 0, 5, 0, 0, 0]]))
    assert len(channel._rewards) == 0

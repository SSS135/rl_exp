import struct
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Dict, List, Tuple, NamedTuple
import numpy as np
import pytest

from mlagents_envs.base_env import AgentId
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage


class AuxRewardSideChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID('ffc44709-4e03-4a6e-a635-60dead4d8ff9'))
        self._rewards: Dict[AgentId, np.ndarray] = dict()
        self.reward_names: List[str] = None

    def on_message_received(self, msg: IncomingMessage) -> None:
        message_id = msg.read_int32()
        assert message_id == 0 or message_id == 1
        if message_id == 0:
            assert self.reward_names is None
            self._process_init_msg(msg)
        elif message_id == 1:
            assert self.reward_names is not None
            self._process_reward_msg(msg)

    def _process_init_msg(self, msg: IncomingMessage):
        count = msg.read_int32()
        self.reward_names = [msg.read_string() for _ in range(count)]

    def _process_reward_msg(self, msg: IncomingMessage):
        agent_id = msg.read_int32()
        reward_index = msg.read_int32()
        reward = msg.read_float32()
        if agent_id not in self._rewards:
            self._rewards[agent_id] = np.zeros(len(self.reward_names), dtype=np.float32)
        self._rewards[agent_id][reward_index] += reward

    def collect_rewards(self, step_agents: np.ndarray) -> np.ndarray:
        # assert len(set(step_agents) - set(self._rewards.keys())) == 0, (set(self._rewards.keys()), set(step_agents))
        rewards = np.zeros((len(step_agents), len(self.reward_names)), dtype=np.float32)
        for i, aid in enumerate(step_agents):
            if aid in self._rewards:
                rewards[i] = self._rewards[aid]
        self._rewards.clear()
        return rewards


def test_AuxRewardSideChannel():
    reward_names = ['a', 'b', 'c', 'd']

    def construct_init_msg():
        m = OutgoingMessage()
        m.write_int32(0)
        m.write_int32(len(reward_names))
        for n in reward_names:
            m.write_string(n)
        return IncomingMessage(m.buffer)

    def construct_reward_msg(agent_id, reward_type, reward):
        m = OutgoingMessage()
        m.write_int32(1)
        m.write_int32(agent_id)
        m.write_int32(reward_type)
        m.write_float32(reward)
        return IncomingMessage(m.buffer)

    channel = AuxRewardSideChannel()
    with pytest.raises(Exception):
        channel.on_message_received(construct_reward_msg(0, 0, 0))

    channel = AuxRewardSideChannel()
    channel.on_message_received(construct_init_msg())
    channel.on_message_received(construct_reward_msg(0, 1, -1))
    channel.on_message_received(construct_reward_msg(7, 2, 2))
    channel.on_message_received(construct_reward_msg(7, 2, 3))

    rewards = channel.collect_rewards(np.array([0, 7]))
    assert np.allclose(rewards, np.asarray([[0, -1, 0, 0], [0, 0, 5, 0]]))
    assert len(channel._rewards) == 0
    assert channel.reward_names == reward_names

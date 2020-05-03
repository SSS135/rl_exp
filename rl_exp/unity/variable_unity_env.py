import os
import random
from functools import partial
from multiprocessing.dummy import Pool
from typing import List, Optional, Tuple

import gym
import gym.spaces
import numpy as np
from mlagents_envs.base_env import BatchedStepResult, AgentGroupSpec, ActionType
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel, EngineConfig
from ppo_pytorch.common.env_factory import NamedVecEnv, Monitor, ObservationNorm, ChannelTranspose
from ppo_pytorch.common.atari_wrappers import FrameStack
from rl_exp.unity.variable_frame_stack import VariableFrameStack
from rl_exp.unity.variable_step_result import VariableStepResult

from .variable_monitor import VariableMonitor

from .variable_env import VariableEnv, VariableVecEnv

from .aux_reward_side_channel import AuxRewardSideChannel
from .simple_unity_env import Command, Message
import multiprocessing as mp


def process_entry(pipe: mp.connection.Connection, env_fn):
    env: VariableUnityEnv = env_fn()
    while True:
        msg: Message = pipe.recv()
        if msg.command == Command.stats:
            pipe.send((env.observation_space, env.action_space))
        elif msg.command == Command.reset:
            pipe.send(env.reset())
        elif msg.command == Command.step:
            pipe.send(env.step(msg.payload))
        elif msg.command == Command.shutdown:
            env.close()
            pipe.send(True)
            return


class VariableUnityEnv(VariableEnv):
    DEFAULT_AGENT_GROUP = 'Player'

    def __init__(self, env_name, visual_observations=False, train_mode=True, *args, base_port=10000, **kwargs):
        self.env_name = env_name
        self.visual_observations = visual_observations
        self.train_mode = train_mode

        self._engine_channel = self._create_engine_channel()
        self._aux_reward_channel = AuxRewardSideChannel()
        self._env = UnityEnvironment(env_name, *args, base_port=base_port, **kwargs,
                                     side_channels=[self._engine_channel, self._aux_reward_channel])
        self._env.reset()

        groups = self._env.get_agent_groups()
        self._agent_group = self.DEFAULT_AGENT_GROUP if self.DEFAULT_AGENT_GROUP in groups else groups[0]
        self.group_spec = self._env.get_agent_group_spec(self._agent_group)
        self.observation_space = self._get_obs_space()
        self.action_space = self._get_action_space()

    def step(self, action) -> VariableStepResult:
        self._env.set_actions(self._agent_group, action)
        self._env.step()
        return self._process_step_result()

    def reset(self) -> VariableStepResult:
        self._env.reset()
        return self._process_step_result()

    def close(self):
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _process_step_result(self) -> VariableStepResult:
        res = self._env.get_step_result(self._agent_group)
        aux_rewards = self._aux_reward_channel.collect_rewards(res.agent_id)
        assert np.allclose(aux_rewards[:, 0], res.reward)
        obs = self._get_vis_obs_list(res)[0] if self.visual_observations else self._get_vector_obs(res)
        if self.visual_observations:
            obs = np.asarray(np.asarray(obs).transpose(0, 3, 1, 2) * 255, dtype=np.uint8)
        return VariableStepResult(obs, aux_rewards, res.done, res.max_step,
                                  res.agent_id, res.reward, None, None)

    def _get_action_space(self):
        shape = self.group_spec.action_shape
        if self.group_spec.action_type == ActionType.CONTINUOUS:
            high = np.ones(shape)
            return gym.spaces.Box(-high, high)
        else:
            return gym.spaces.MultiDiscrete(shape)

    def _get_obs_space(self):
        if self.visual_observations:
            obs_shape = self._get_vis_obs_shape()
            zeros = np.zeros((obs_shape[2], obs_shape[0], obs_shape[1]))
            return gym.spaces.Box(zeros, zeros + 255, dtype=np.uint8)
        else:
            obs_size = self._get_vec_obs_size()
            ones = np.ones(obs_size)
            return gym.spaces.Box(-ones, ones, dtype=np.float32)

    def _create_engine_channel(self):
        engine_channel = EngineConfigurationChannel()
        engine_config = EngineConfig(80, 80, 1, 4.0, 30 * 4) if self.train_mode else EngineConfig(1280, 720, 1, 1.0, 60)
        engine_channel.set_configuration(engine_config)
        return engine_channel

    def _get_vis_obs_shape(self) -> Optional[Tuple]:
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 3:
                return shape
        return None

    def _get_vec_obs_size(self) -> int:
        result = 0
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 1:
                result += shape[0]
        return result

    def _get_vis_obs_list(self, step_result: BatchedStepResult) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vector_obs(self, step_result: BatchedStepResult) -> np.ndarray:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)


def make_env(env_path, visual_observations, stacked_frames, no_graphics):
    while True:
        worker_id = random.randrange(50000)
        try:
            env = VariableUnityEnv(env_path, worker_id=worker_id,
                                   visual_observations=visual_observations, no_graphics=no_graphics)
            break
        except UnityWorkerInUseException:
            print(f'Worker {worker_id} already in use')
    env = VariableMonitor(env)
    if stacked_frames > 1:
        env = VariableFrameStack(env, stacked_frames)
    return env


class VariableUnityVecEnv(VariableVecEnv):
    def __init__(self, env_path, num_envs, visual_observations=False, stacked_frames=1):
        self.env_path = env_path
        self.visual_observations = visual_observations
        self.stacked_frames = stacked_frames
        self.env_name = os.path.basename(os.path.split(os.path.normpath(env_path))[0])

        self._processes: List[mp.Process] = []
        self._pipes: List[mp.connection.Connection] = []
        while len(self._processes) < num_envs:
            self._create_process()
        self.observation_space, self.action_space = self._send_message(Command.stats)[0]
        self._actors_per_env = None

    @property
    def num_envs(self):
        return len(self._processes)

    def step(self, actions: np.ndarray) -> VariableStepResult:
        actions = np.split(np.asarray(actions), self._actors_per_env)
        data = self._send_message(Command.step, actions)
        return self._data_to_step_result(data)

    def reset(self) -> VariableStepResult:
        data = self._send_message(Command.reset)
        return self._data_to_step_result(data)

    def _data_to_step_result(self, data: List[VariableStepResult]) -> VariableStepResult:
        self._actors_per_env = [len(data[0].agent_id)]
        for x in data[1:-1]:
            self._actors_per_env.append(len(x.agent_id) + self._actors_per_env[-1])

        data = [(x.obs, x.rewards, x.done, x.max_step, x.agent_id, x.true_reward, x.total_true_reward, x.episode_length) for x in data]
        return VariableStepResult(*[np.concatenate(x, 0) for x in zip(*data)])

    def _create_process(self):
        agent_id = len(self._processes)
        parent_conn, child_conn = mp.Pipe()
        env_factory = partial(self._get_env_factory(), no_graphics=agent_id > 0)
        proc = mp.Process(name=f'agent {agent_id}', target=process_entry, args=(child_conn, env_factory))
        proc.start()
        self._processes.append(proc)
        self._pipes.append(parent_conn)

    def _get_env_factory(self):
        return partial(make_env, self.env_path, self.visual_observations, self.stacked_frames)

    def _send_message(self, command: Command, payload: Optional[List] = None) -> List:
        if payload is None:
            payload = len(self._processes) * [None]
        for pipe, payload in zip(self._pipes, payload):
            pipe.send(Message(command, payload))
        return [pipe.recv() for pipe in self._pipes]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._send_message(Command.shutdown)
        for proc in self._processes:
            proc.join(30)
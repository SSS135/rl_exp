import os
import random
from functools import partial
from multiprocessing.dummy import Pool
from typing import List, Optional

import gym
import gym.spaces
import numpy as np
from mlagents_envs.base_env import BatchedStepResult, AgentGroupSpec, ActionType
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel, EngineConfig
from ppo_pytorch.common.env_factory import NamedVecEnv, Monitor, ObservationNorm, ChannelTranspose
from ppo_pytorch.common.atari_wrappers import FrameStack
from .variable_monitor import VariableMonitor

from .variable_env import VariableEnv

from .aux_reward_side_channel import AuxRewardSideChannel
from .simple_unity_env import Command, Message, process_entry
import multiprocessing as mp


DEFAULT_AGENT_GROUP = 'Player'


class VariableUnityEnv(VariableEnv):
    def __init__(self, env_name, visual_observations=False, train_mode=True, *args, base_port=10000, **kwargs):
        self.env_name = env_name
        self.visual_observations = visual_observations
        self.train_mode = train_mode

        self._env = UnityEnvironment(env_name, *args, base_port=base_port, **kwargs,
                                     side_channels=[self._create_engine_channel(), AuxRewardSideChannel()])
        self._env.reset()

        groups = self._env.get_agent_groups()
        self._agent_group = DEFAULT_AGENT_GROUP if DEFAULT_AGENT_GROUP in groups else groups[0]
        agent_spec = self._env.get_agent_group_spec(self._agent_group)
        self.observation_space = self._get_observation_space(agent_spec)
        self.action_space = self._get_action_space(agent_spec)

    def _create_engine_channel(self):
        engine_channel = EngineConfigurationChannel()
        engine_config = EngineConfig(80, 80, 1, 4.0, 30 * 4) if self.train_mode else EngineConfig(1280, 720, 1, 1.0, 60)
        engine_channel.set_configuration(engine_config)
        return engine_channel

    def step(self, action) -> BatchedStepResult:
        self._env.set_actions(self._agent_group, action)
        self._env.step()
        res = self._env.get_step_result(self._agent_group)
        return self._process_brain_info(res)

    def reset(self) -> BatchedStepResult:
        self._env.reset()
        res = self._env.get_step_result(self._agent_group)
        return self._process_brain_info(res)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _process_brain_info(self, env_state: BatchedStepResult) -> BatchedStepResult:
        obs = env_state.obs[self._obs_index]
        if self.visual_observations:
            obs = np.asarray(np.asarray(obs).transpose(0, 3, 1, 2) * 255, dtype=np.uint8)
        return env_state._replace(obs=obs)

    def _get_action_space(self, spec: AgentGroupSpec):
        shape = spec.action_shape
        if spec.action_type == ActionType.CONTINUOUS:
            assert isinstance(spec.action_type, int)
            high = np.ones(shape)
            return gym.spaces.Box(-high, high)
        else:
            return gym.spaces.MultiDiscrete(shape)

    def _get_observation_space(self, spec: AgentGroupSpec):
        obs_shape, self._obs_index = next(filter(lambda s, i: len(s) == (4 if self.visual_observations else 2),
                                                 spec.observation_shapes))
        if self.visual_observations:
            zeros = np.zeros(obs_shape)
            return gym.spaces.Box(zeros, zeros + 255, dtype=np.uint8)
        else:
            ones = np.ones(obs_shape)
            return gym.spaces.Box(-ones, ones, dtype=np.float32)

    def close(self):
        self._env.close()


def make_env(env_path, visual_observations, no_graphics):
    while True:
        worker_id = random.randrange(50000)
        try:
            env = VariableUnityEnv(env_path, worker_id=worker_id,
                                   visual_observations=visual_observations, no_graphics=no_graphics)
            break
        except UnityWorkerInUseException:
            print(f'Worker {worker_id} already in use')
    env = VariableMonitor(env)
    # if stacked_frames > 1:
    #     assert visual_observations
    #     env = FrameStack(env, stacked_frames)
    return env


class VariableUnityVecEnv:
    def __init__(self, env_path, visual_observations=False, observation_norm=False,
                 stacked_frames=1):
        self.env_path = env_path
        self.visual_observations = visual_observations
        env_name = os.path.basename(os.path.split(os.path.normpath(env_path))[0])
        self.observation_norm = observation_norm
        self.stacked_frames = stacked_frames
        self.env_name = env_name
        self.num_actors = None

        self._processes: List[mp.Process] = []
        self._pipes: List[mp.connection.Connection] = []

        self._create_process()
        self.observation_space, self.action_space, self.actors_per_env = self._send_message(Command.stats)[0]

    def _create_process(self):
        agent_id = len(self._processes)
        parent_conn, child_conn = mp.Pipe()
        env_factory = partial(self.get_env_factory(), no_graphics=False)
        proc = mp.Process(name=f'agent {agent_id}', target=process_entry, args=(child_conn, env_factory))
        proc.start()
        self._processes.append(proc)
        self._pipes.append(parent_conn)

    def get_env_factory(self):
        return partial(make_env, self.env_path, self.observation_norm, self.visual_observations, self.stacked_frames)

    def set_num_actors(self, num_actors):
        assert num_actors >= len(self._processes) * self.actors_per_env
        assert num_actors % self.actors_per_env == 0, (num_actors, self.actors_per_env)

        self.num_actors = num_actors
        num_envs = self.num_actors // self.actors_per_env
        while len(self._processes) < num_envs:
            self._create_process()

    def step(self, actions):
        actions = np.split(np.asarray(actions), len(self._processes))
        data = self._send_message(Command.step, actions)
        return [np.stack(x, 0) for x in zip(*data)]

    def reset(self):
        data = self._send_message(Command.reset)
        return np.stack(data, 0)

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


class UnityVecEnv(NamedVecEnv):
    def __init__(self, env_path, parallel='thread', visual_observations=False, observation_norm=False,
                 stacked_frames=1, train_mode=True):
        self.env_path = env_path
        self.visual_observations = visual_observations
        self.train_mode = train_mode
        env_name = os.path.basename(os.path.split(os.path.normpath(env_path))[0])
        self.observation_norm = observation_norm
        self.stacked_frames = stacked_frames
        self.env_name = env_name
        self.parallel = parallel
        self.envs = None
        self.num_envs = None
        self._pool = None

        env: UnityEnv = self.get_env_factory()()
        data = env.reset()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._initial_num_players = len(data.agent_id)
        env.close()

    def get_env_factory(self):
        def make(env_path, observation_norm, visual_observations, stacked_frames, train_mode):
            while True:
                worker_id = random.randrange(50000)
                try:
                    env = UnityEnv(env_path, worker_id=worker_id, visual_observations=visual_observations,
                                   train_mode=train_mode)
                    break
                except UnityWorkerInUseException:
                    print(f'Worker {worker_id} already in use')
            env = Monitor(env)
            if visual_observations:
                env = ChannelTranspose(env)
            if stacked_frames > 1:
                assert visual_observations
                env = FrameStack(env, 4)
            if observation_norm:
                env = ObservationNorm(env)
            return env

        return partial(make, self.env_path, self.observation_norm, self.visual_observations, self.stacked_frames, self.train_mode)

    def set_num_actors(self, num_actors):
        if self.envs is not None:
            for env in self.envs:
                env.close()
        self.num_envs = num_actors
        env_factory = self.get_env_factory()
        self._pool = Pool(self.num_envs)
        self.envs = self._pool.map(lambda _: env_factory(), range(num_actors))

    def step(self, actions) -> BatchedStepResult:
        actions = np.split(np.asarray(actions), len(self.envs))
        data = self._pool.starmap(lambda env, a: env.step(a), zip(self.envs, actions))
        return BatchedStepResult._make([np.concatenate(x, 0) for x in zip(*[tuple(t) for t in data])])

    def reset(self) -> BatchedStepResult:
        data = self._pool.starmap(lambda env: env.reset(), zip(self.envs))
        return BatchedStepResult._make([np.concatenate(x, 0) for x in zip(*[tuple(t) for t in data])])

    def __enter__(self):
        self._pool.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.map(lambda env: env.close(), self.envs)
        self._pool.__exit__()


# class SimplifiedUnityVecEnv(UnityVecEnv):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.num_players = self._initial_num_players
#         self._index_to_id = {}
#         self._prev_ids = []
#
#     def set_num_envs(self, num_envs):
#         super().set_num_envs(num_envs // self.num_players)
#
#     def step(self, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
#         res = super().step(actions)
#         if len(self._prev_ids) != len(res.agent_id):
#
#         return res.obs, res.reward, res.done, [dict() for _ in range(self.num_players)]
#
#     def reset(self) -> np.ndarray:
#         res = super().reset()
#         self._index_to_id = {index: id for id, index in res.agent_id_to_index.items()}
#         self._prev_ids = res.agent_id
#         return res.obs
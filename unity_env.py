import os
import random
import time
from functools import partial
from multiprocessing.dummy import Pool

import gym
import gym.spaces
import numpy as np
from ppo_pytorch.common.multiplayer_env import MultiplayerEnv
from mlagents.envs.exception import UnityWorkerInUseException
from ppo_pytorch.common.env_factory import NamedVecEnv, SimpleVecEnv, Monitor, ObservationNorm, ChannelTranspose
from mlagents.envs.environment import UnityEnvironment, BrainInfo, BrainParameters, AllBrainInfo


class UnityEnv(MultiplayerEnv):
    def __init__(self, env_name, visual_observations=False, train_mode=True, *args, **kwargs):
        self.visual_observations = visual_observations
        self.train_mode = train_mode
        self.env = UnityEnvironment(env_name, *args, base_port=10000, **kwargs)
        brain = next(iter(self.env.reset(train_mode=self.train_mode).values()))
        assert self.env.number_external_brains == 1
        brain_params = self.env.external_brains[self.env.external_brain_names[0]]
        self.observation_space = self._get_observation_space(brain_params, brain)
        self.action_space = self._get_action_space(brain_params, brain)
        self.reward_range = (-1, 1)
        super().__init__(len(brain.agents))

    def step(self, action, *args, **kwargs):
        state = self.env.step(action, *args, **kwargs)
        return self._process_brain_info(state)

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs, train_mode=self.train_mode)
        return self._process_brain_info(state)[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _process_brain_info(self, env_state: AllBrainInfo):
        assert len(env_state) == 1, env_state
        brain: BrainInfo = next(iter(env_state.values()))
        if self.visual_observations:
            states = np.asarray(np.asarray(brain.visual_observations[0]) * 255, dtype=np.uint8)
        else:
            states = brain.vector_observations
        return states, brain.rewards, brain.local_done, [{}] * self.num_players

    def _get_action_space(self, brain_params: BrainParameters, info: BrainInfo):
        type, size = brain_params.vector_action_space_type, brain_params.vector_action_space_size
        return self._create_space(type, size)

    def _get_observation_space(self, brain_params: BrainParameters, info: BrainInfo):
        if self.visual_observations:
            assert brain_params.number_visual_observations == 1
            zeros = np.zeros(np.shape(info.visual_observations)[-3:])
            return gym.spaces.Box(zeros, zeros + 255, dtype=np.uint8)
        else:
            size = brain_params.vector_observation_space_size * brain_params.num_stacked_vector_observations
            return self._create_space('continuous', [size])

    def _create_space(self, type, size):
        if type == 'continuous':
            assert len(size) == 1
            high = np.ones(size[0])
            return gym.spaces.Box(-high, high)
        else:
            return gym.spaces.MultiDiscrete(size) if len(size) > 1 else gym.spaces.Discrete(size[0])

    def close(self):
        self.env.close()


class UnityVecEnv(NamedVecEnv):
    def __init__(self, env_path, parallel='thread', visual_observations=False, observation_norm=False, train_mode=True):
        self.env_path = env_path
        self.visual_observations = visual_observations
        self.train_mode = train_mode
        env_name = os.path.basename(os.path.split(os.path.normpath(env_path))[0])
        self.observation_norm = observation_norm
        self.env_name = env_name
        self.parallel = parallel
        self.envs = None
        self.num_envs = None
        self.pool = None

        env: UnityEnv = self.get_env_factory()()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_players = env.num_players
        env.close()

    def get_env_factory(self):
        def make(env_path, observation_norm, visual_observations, train_mode=True):
            while True:
                worker_id = random.randrange(50000)
                try:
                    env = UnityEnv(env_path, worker_id=worker_id, visual_observations=visual_observations,
                                   train_mode=train_mode)
                    break
                except UnityWorkerInUseException:
                    print(f'Worker {worker_id} already in use')
            env = Monitor(env)
            if self.visual_observations:
                env = ChannelTranspose(env)
            if observation_norm:
                env = ObservationNorm(env)
            return env

        return partial(make, self.env_path, self.observation_norm, self.visual_observations, self.train_mode)

    def set_num_envs(self, num_envs):
        assert num_envs % self.num_players == 0, (num_envs, self.num_players)
        if self.envs is not None:
            for env in self.envs:
                env.close()
        self.num_envs = num_envs
        env_factory = self.get_env_factory()
        self.pool = Pool(self.num_envs // self.num_players)
        self.envs = self.pool.map(lambda _: env_factory(), range(num_envs // self.num_players))

    def step(self, actions):
        actions = np.split(np.asarray(actions), len(self.envs))
        data = self.pool.starmap(lambda env, a: env.step(a), zip(self.envs, actions))
        return [np.concatenate(x, 0) for x in zip(*data)]

    def reset(self):
        return np.concatenate(self.pool.map(lambda env: env.reset(), self.envs), 0)

    def __enter__(self):
        self.pool.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.map(lambda env: env.close(), self.envs)
        self.pool.__exit__()

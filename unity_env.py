import os
import random
import time
from functools import partial

import gym
import gym.spaces
import numpy as np
from mlagents.envs.exception import UnityWorkerInUseException
from ppo_pytorch.common.env_factory import NamedVecEnv, SimpleVecEnv, Monitor, ObservationNorm
from mlagents.envs.environment import UnityEnvironment, BrainInfo, BrainParameters, AllBrainInfo


class UnityEnv(gym.Env):
    def __init__(self, env_name, *args, **kwargs):
        self.env = UnityEnvironment(env_name, *args, base_port=10000, **kwargs)
        self.env.reset()
        assert self.env.number_external_brains == 1, self.env.number_external_brains
        brain_params = self.env.brains[self.env.external_brain_names[0]]
        self.observation_space = self._get_observation_space(brain_params)
        self.action_space = self._get_action_space(brain_params)
        self.reward_range = (-1, 1)

    def step(self, action, *args, **kwargs):
        state = self.env.step(self._process_action(action), *args, **kwargs)
        return self._next(state)

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        return self._next(state)[0]

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _next(self, env_state: AllBrainInfo):
        assert len(env_state) == 1, env_state
        brain: BrainInfo = next(iter(env_state.values()))
        assert len(brain.agents) == 1, brain.agents
        states = brain.vector_observations
        return self._process_state(states[0]), brain.rewards[0], brain.local_done[0], {}

    def _get_action_space(self, brain_params: BrainParameters):
        assert len(brain_params.vector_action_space_size) == 1
        return self._create_space(brain_params.vector_action_space_type, brain_params.vector_action_space_size[0])

    def _get_observation_space(self, brain_params: BrainParameters):
        return self._create_space('continuous', brain_params.vector_observation_space_size * brain_params.num_stacked_vector_observations)

    def _process_action(self, action):
        return action

    def _process_state(self, state):
        return state

    def _create_space(self, type, size):
        if type == 'continuous':
            high = np.ones(size)
            return gym.spaces.Box(-high, high)
        else:
            return gym.spaces.Discrete(size)

    def close(self):
        self.env.close()


class UnityVecEnv(SimpleVecEnv):
    def __init__(self, env_path, observation_norm=False, parallel='thread'):
        self.env_path = env_path
        env_name = os.path.basename(os.path.split(os.path.normpath(env_path))[0])
        super().__init__(env_name, observation_norm, parallel)

    def get_env_fn(self):
        def make(env_path, observation_norm):
            while True:
                worker_id = random.randrange(50000)
                try:
                    env = UnityEnv(env_path, worker_id=worker_id)
                    break
                except UnityWorkerInUseException:
                    print(f'Worker {worker_id} already in use')
            env = Monitor(env)
            if observation_norm:
                env = ObservationNorm(env)
            return env

        return partial(make, self.env_path, self.observation_norm)


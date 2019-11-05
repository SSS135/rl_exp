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
from ppo_pytorch.common.env_factory import NamedVecEnv, SimpleVecEnv, Monitor, ObservationNorm
from mlagents.envs.environment import UnityEnvironment, BrainInfo, BrainParameters, AllBrainInfo


class UnityEnv(MultiplayerEnv):
    def __init__(self, env_name, *args, **kwargs):
        self.env = UnityEnvironment(env_name, *args, base_port=10000, **kwargs)
        brain = next(iter(self.env.reset().values()))
        assert self.env.number_external_brains == 1
        brain_params = self.env.external_brains[self.env.external_brain_names[0]]
        self.observation_space = self._get_observation_space(brain_params)
        self.action_space = self._get_action_space(brain_params)
        self.reward_range = (-1, 1)
        super().__init__(len(brain.agents))

    def step(self, action, *args, **kwargs):
        state = self.env.step(self._process_action(action), *args, **kwargs)
        return self._process_brain_info(state)

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        return self._process_brain_info(state)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _process_brain_info(self, env_state: AllBrainInfo):
        assert len(env_state) == 1, env_state
        brain: BrainInfo = next(iter(env_state.values()))
        states = [self._process_state(s) for s in brain.vector_observations]
        return states, brain.rewards, brain.local_done, [{}] * self.num_players

    def _get_action_space(self, brain_params: BrainParameters):
        type, size = brain_params.vector_action_space_type, brain_params.vector_action_space_size
        return self._create_space(type, size)

    def _get_observation_space(self, brain_params: BrainParameters):
        return self._create_space('continuous', [brain_params.vector_observation_space_size * brain_params.num_stacked_vector_observations])

    def _process_action(self, action):
        return action

    def _process_state(self, state):
        return state

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
    def __init__(self, env_path, observation_norm=False, parallel='thread'):
        self.env_path = env_path
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

    def set_num_envs(self, num_envs):
        assert num_envs % self.num_players == 0, (num_envs, self.num_players)
        if self.envs is not None:
            for env in self.envs:
                env.close()
        self.num_envs = num_envs
        env_factory = self.get_env_factory()
        self.envs = [env_factory() for _ in range(num_envs // self.num_players)]
        # self.pool = Pool(self.num_envs // self.num_players)

    def step(self, actions):
        actions = np.split(np.asarray(actions), len(self.envs))
        return self._cat_data(env.step(a) for env, a in zip(self.envs, actions))

    def reset(self):
        return self._cat_data(env.reset() for env in self.envs)[0]

    def _cat_data(self, data):
        return [np.concatenate(x, 0) for x in zip(*data)]

    # def __enter__(self):
    #     self.pool.__enter__()
    #     return self
    #
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     self.pool.__exit__()
    #     for env in self.envs:
    #         env.close()

import os
import random
from functools import partial
from multiprocessing.dummy import Pool

import numpy as np
from gym_unity.envs import UnityEnv
from mlagents_envs.exception import UnityWorkerInUseException
from ppo_pytorch.common.atari_wrappers import FrameStack
from ppo_pytorch.common.env_factory import NamedVecEnv, Monitor, ObservationNorm, ChannelTranspose


class UnityVecEnv(NamedVecEnv):
    def __init__(self, env_path, parallel='process', visual_observations=False, observation_norm=False,
                 stacked_frames=1):
        self.env_path = env_path
        self.visual_observations = visual_observations
        env_name = os.path.basename(os.path.split(os.path.normpath(env_path))[0])
        self.observation_norm = observation_norm
        self.stacked_frames = stacked_frames
        self.env_name = env_name
        self.parallel = parallel
        self.envs = None
        self.num_actors = None
        self._pool = None

        env: UnityEnv = self.get_env_factory()(no_graphics=True)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.actors_per_env = env.number_agents
        env.close()

    def get_env_factory(self):
        def make(env_path, observation_norm, visual_observations, stacked_frames, no_graphics=False):
            while True:
                worker_id = random.randrange(50000)
                try:
                    env = UnityEnv(env_path, worker_id=worker_id, no_graphics=no_graphics,
                                   uint8_visual=visual_observations, use_visual=visual_observations)
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

        return partial(make, self.env_path, self.observation_norm, self.visual_observations, self.stacked_frames)

    def set_num_actors(self, num_actors):
        assert num_actors % self.actors_per_env == 0, (num_actors, self.actors_per_env)
        if self.envs is not None:
            for env in self.envs:
                env.close()
        self.num_actors = num_actors
        env_factory = self.get_env_factory()
        self._pool = Pool(self.num_actors // self.actors_per_env)
        self.envs = self._pool.map(lambda i: env_factory(no_graphics=i != 0 and not self.visual_observations),
                                   range(num_actors // self.actors_per_env))

    def step(self, actions):
        actions = np.split(np.asarray(actions), len(self.envs))
        data = self._pool.starmap(lambda env, a: env.step(a), zip(self.envs, actions))
        return [np.stack(x, 0) for x in zip(*data)]

    def reset(self):
        return np.stack(self._pool.map(lambda env: env.reset(), self.envs), 0)

    def __enter__(self):
        self._pool.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.map(lambda env: env.close(), self.envs)
        self._pool.__exit__()
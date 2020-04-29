from collections import namedtuple
from itertools import count

import gym
import numpy as np
from ppo_pytorch.experimental.parallel_env import ParallelEnv

Sample = namedtuple('EnvSample', 'states, actions, rewards, next_states, values, dones, meta')


class EnvSampler:
    def __init__(self,
                 env: ParallelEnv or gym.Env,
                 actor,
                 gae_gamma=0.99,
                 gae_lambda=0.95):
        self.env = env
        self.actor = actor
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        if not isinstance(self.env, ParallelEnv):
            self.env = ParallelEnv(lambda: self.env, 1)
        self.states = self.env.reset()
        self.actions, self.values, self.meta = self.actor.act(self.states)

    def sample(self, steps: int or None) -> Sample:
        assert steps is not None or self.env.num_envs == 1

        samples = []
        for _ in (count() if steps is None else range(steps)):
            cur_states = self.states
            self.states, rewards, dones, _ = self.env.step(self.actions)
            self.actor.complete_episodes(dones)
            samples.append((cur_states, self.actions, rewards, self.states, self.values, dones, self.meta))
            self.actions, self.values, self.meta = self.actor.act(self.states)
            if steps is None and dones[0]:
                break

        states, actions, rewards, next_states, values, dones, meta = list(map(list, zip(*samples)))
        rewards, values, dones = np.asarray(rewards), np.asarray(values + [self.values]), np.asarray(dones)

        return Sample(states, actions, rewards, next_states, values, dones, meta)

import gym
import numpy as np


class ParallelEnv:
    def __init__(self, env_factory, num_envs):
        self.env_factory = env_factory
        self.num_envs = num_envs

        if isinstance(self.env_factory, str):
            env_name = self.env_factory
            self.env_factory = lambda: gym.make(env_name)
        self.envs = [self.env_factory() for _ in range(self.num_envs)]
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.reward_range = self.envs[0].reward_range

    def step(self, inputs):
        inputs = np.asarray(inputs)
        assert inputs.shape[0] == self.num_envs
        outs = []
        for env, input in zip(self.envs, inputs):
            state, r, done, inf = env.step(input)
            if done:
                state = env.reset()
            outs.append((state, r, done, inf))
        states, rewards, dones, infos = list(zip(*outs))
        return np.array(states), np.array(rewards), np.array(dones), list(infos)

    def reset(self):
        return np.array([e.reset() for e in self.envs])

    def close(self):
        for env in self.envs:
            env.close()

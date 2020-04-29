# https://github.com/rarilurelo/pytorch_a3c/blob/master/wrapper_env.py

import gym
import numpy as np
from gym.spaces import Box


class AtariEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        print(env.observation_space)
        self.observation_space = Box(low=0, high=1, shape=env.observation_space.shape[:-1])

    def _observation(self, observation):
        x = np.squeeze(observation, -1)
        if x.ndim == 3:
            print(x.shape)
            x = x.transpose(2, 0, 1)
        print(x.shape)
        return x

    # def step(self, action):
    #     o, r, done, env_info = self.env.step(action)
    #     o = self._preprocess_obs(o)
    #     return o, r, done, env_info
    #
    # def reset(self):
    #     o = self.env.reset()
    #     o = self._preprocess_obs(o)
    #     return o
    #
    # def render(self):
    #     self.env.render()
    #
    # def seed(self, s):
    #     self.env.seed(s)
    #
    # @property
    # def action_space(self):
    #     return self.env.action_space
    #
    # @property
    # def observation_space(self):
    #     return self._observation_space
    #
    # @property
    # def reward_range(self):
    #     return self.env.reward_range
    #
    # def _preprocess_obs(self, obs):
    #     assert obs.ndim == 3  # (height, width, channel)
    #     img = Image.fromarray(obs)
    #     img = img.crop((0, 25, 160, 160))
    #     img = img.resize(self.input_shape).convert('L')  # resize and convert to grayscale
    #     processed_observation = np.array(img)
    #     assert processed_observation.shape == self.input_shape
    #     return self._to_float(processed_observation)
    #
    # def _to_float(self, data):
    #     """
    #     int to float
    #     """
    #     processed_data = data.astype(np.float32) / 255.
    #     return processed_data
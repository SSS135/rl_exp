from typing import List

import gym
import numpy as np


class MultiSpace(gym.Space):
    def __init__(self, spaces: List[gym.Space], normalize=False):
        self.train_weights = np.ones(len(spaces))
        if normalize:
            self.train_weights = self.train_weights / self.train_weights.sum()
        self.spaces = spaces

    def sample(self):
        raise NotImplementedError

    def contains(self, x):
        raise NotImplementedError

    def __repr__(self):
        return f'MultiSpace{self.spaces}'

    def __str__(self):
        return self.__repr__()

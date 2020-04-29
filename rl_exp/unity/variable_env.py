import gym
import numpy as np
from .variable_step_result import VariableStepResult


class VariableEnv:
    action_space: gym.Space = None
    observation_space: gym.Space = None

    def step(self, action: np.ndarray) -> VariableStepResult:
        raise NotImplementedError

    def reset(self) -> VariableStepResult:
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed=None):
        return

    @property
    def unwrapped(self):
        return self

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False
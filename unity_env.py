import time

import gym
import gym.spaces
import numpy as np
from unityagents import UnityEnvironment, BrainInfo, BrainParameters


class UnityEnv(gym.Env):
    def __init__(self, env_name, use_states=True, *args, **kwargs):
        self.use_states = use_states
        self.env = UnityEnvironment(env_name, *args, **kwargs)
        time.sleep(2) # FIXME: otherwise env.reset may freeze
        assert self.env.number_brains == 1
        brain_params: BrainParameters = self.env.brains[self.env.brain_names[0]]
        self.observation_space = self._get_observation_space(brain_params)
        self.action_space = self._get_action_space(brain_params)
        self.reward_range = (-1, 1)

    def step(self, action, *args, **kwargs):
        return self._step(action, *args, **kwargs)

    def reset(self, *args, **kwargs):
        return self._reset(*args, **kwargs)

    def _step(self, action, *args, **kwargs):
        state = self.env.step(self._process_action(action), *args, **kwargs)
        return self._next(state)

    def _reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        return self._next(state)[0]

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _next(self, env_state):
        assert len(env_state) == 1
        brain: BrainInfo = next(iter(env_state.values()))
        assert len(brain.agents) == 1
        states = brain.states if self.use_states else brain.observations
        return self._process_state(states[0]), brain.rewards[0], brain.local_done[0], {}

    def _get_action_space(self, brain_params: BrainParameters):
        return self._create_space(brain_params.action_space_type, brain_params.action_space_size)

    def _get_observation_space(self, brain_params: BrainParameters):
        return self._create_space(brain_params.state_space_type, brain_params.state_space_size)

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


# class MultiplayerUnityEnv(MultiplayerEnv):
#     def __init__(self, env_name, use_states=True):
#         self.use_states = use_states
#         self.env = UnityEnvironment(env_name)
#         assert self.env.number_brains == 1
#
#         brain_params: BrainParameters = self.env.brains[self.env.brain_names[0]]
#         brain_info: BrainInfo = self.env.reset()
#
#         super().__init__(len(brain_info.agents))
#         #self.observation_space = gym.spaces.Box brain_params.state_space_type == 'continuous'
#
#     def _step(self, action):
#         return self._next(self.env.step(action))
#
#     def _reset(self):
#         return self._next(self.env.reset())[0]
#
#     def _next(self, env_state):
#         assert len(env_state) == 1
#         brain: BrainInfo = next(iter(env_state.values()))
#         d = np.array(brain.local_done)
#         assert np.all(d) or np.all(1 - d)
#         states = brain.states if self.use_states else brain.observations
#         return states, brain.rewards, brain.local_done[0], {}


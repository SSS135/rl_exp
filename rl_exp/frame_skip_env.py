from collections import deque, Callable

import gym
import numpy as np
from gym.spaces import Box


class FrameSkipEnv:
    def __init__(self, env: Callable or gym.Env, frame_skip, merged_frames, frame_merger='cat'):
        if callable(env):
            env = env()

        self.env = env
        self.frame_skip = frame_skip
        self.frame_merger = frame_merger
        self.episode_len = 0
        self.merged_frames = merged_frames = (frame_skip + 1) if merged_frames is None else merged_frames
        self.prev_observations = None

        if isinstance(self.env, str):
            self.env = gym.make(env)

        l = env.observation_space.low
        h = env.observation_space.high
        if self.frame_merger == 'last':
            self.frame_merger = lambda f: f[-1]
        elif self.frame_merger == 'cat':
            self.frame_merger = lambda f: np.concatenate(f)
            self.observation_space = Box(np.concatenate([l]*merged_frames), np.concatenate([h]*merged_frames))
        elif self.frame_merger == 'stack':
            self.frame_merger = lambda f: np.stack(f, axis=0)
            self.observation_space = Box(np.stack([l]*merged_frames), np.stack([h]*merged_frames))

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def step(self, input):
        outs = []
        iters = self.frame_skip + 1
        for i in range(iters):
            s, r, d, i = self.env.step(input)
            outs.append((r, d, i))
            self.episode_len += 1
            if d:
                break
        self.prev_observations.append(s)
        rewards, dones, infos = list(zip(*outs))
        return self.frame_merger(self.prev_observations), np.sum(rewards, axis=0), dones[-1], infos[-1]

    def reset(self):
        self.episode_len = 0
        state = self.env.reset()
        self.prev_observations = deque([state]*self.merged_frames, self.merged_frames)
        return self.frame_merger(self.prev_observations)


class ConditionalFrameSkipEnv(FrameSkipEnv):
    def __init__(self, condition_check, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_check = condition_check

    def step(self, input):
        outs = []
        iters = self.frame_skip + 1
        iter = 0
        while iter < iters:
            s, r, d, i = self.env.step(input)
            outs.append((r, d, i))
            self.episode_len += 1
            if d:
                break
            if self.condition_check(s):
                iter += 1
        self.prev_observations.append(s)
        rewards, dones, infos = list(zip(*outs))
        return self.frame_merger(self.prev_observations), np.sum(rewards, axis=0), dones[-1], infos[-1]
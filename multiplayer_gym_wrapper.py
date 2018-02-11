import pprint
from itertools import count

import gym
from mylib.ppo_pytorch.common.tensorboard_env_logger import TensorboardEnvLogger
from ppo_pytorch.experimental.frame_skip_env import FrameSkipEnv
from ppo_pytorch.experimental.observation_norm_env import ObservationNormalizer


class MultiplayerGymWrapper:
    def __init__(self,
                 rl_alg_factories,
                 env_factory,
                 frame_skip=0,
                 merged_frames=1,
                 frame_merger='cat',
                 log_time_interval=5,
                 log_path=None,
                 observation_norm=True,
                 reward_std_episodes=100,
                 logged_player_index=0,):
        self._init_args = locals()
        self.rl_alg_factories = rl_alg_factories
        self.env_factory = env_factory
        self.frame_skip = frame_skip
        self.merged_frames = merged_frames
        self.frame_merger = frame_merger
        self.observation_norm = observation_norm
        self.logged_player_index = logged_player_index
        self.frame = 0
        self.done = True

        self.env = self._make_env()
        self.states = None
        self.rl_algs = [f(self.env.observation_space, self.env.action_space, log_time_interval=log_time_interval)
                        for f in rl_alg_factories]
        assert all(a.num_actors == 1 for a in self.rl_algs)
        assert len(rl_alg_factories) == self.env.num_players

        if log_path is not None:
            name = str([type(a).__name__ for a in self.rl_algs])
            self.log = TensorboardEnvLogger(name, log_path, 1, log_time_interval, reward_std_episodes)
        else:
            self.log = None

    def _make_env(self):
        env = gym.make(self.env_factory) if isinstance(self.env_factory, str) else self.env_factory
        if self.observation_norm:
            env = ObservationNormalizer(env)
        if self.frame_skip > 0 or self.merged_frames > 1:
            env = FrameSkipEnv(env, self.frame_skip, merged_frames=self.merged_frames, frame_merger=self.frame_merger)
        return env

    def __enter__(self):
        assert self.log is not None
        self.log.__enter__()
        self.log.add_text('GymWrapper', pprint.pformat(self._init_args))
        self.rl_algs[self.logged_player_index].logger = self.log

    def __exit__(self, *args, **kwargs):
        self.log.__exit__(*args, **kwargs)

    def step(self, always_log=False):
        if self.done:
            self.states = self.env.reset()
        actions = [alg.eval(s, [self.env]) for s, alg in zip(self.states, self.rl_algs)]
        self.states, rewards, self.done, _ = self.env.step(actions)

        for r, alg in zip(rewards, self.rl_algs):
            alg.reward(r)
            alg.finish_episodes(self.done)

        if self.log is not None:
            self.log.step(rewards[self.logged_player_index], self.done, always_log)
        self.frame += 1

    def train(self, max_frames):
        for _ in count():
            self.step(self.frame + 1 >= max_frames)
            if self.frame >= max_frames:
                break

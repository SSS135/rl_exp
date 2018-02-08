from gym.spaces import Box

from .online_normalization import OnlineNormalizer


class ObservationNormalizer:
    def __init__(self, env, eps=0.01, absmax=5):
        assert isinstance(env.observation_space, Box)
        self.env = env
        self.eps = eps
        self.observation_normalizer = OnlineNormalizer(eps, absmax)

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def step(self, input):
        if hasattr(self.env, 'num_players'):
            states, r, d, i = self.env.step(input)
            states = [self.observation_normalizer(s) for s in states]
            return states, r, d, i
        else:
            s, r, d, i = self.env.step(input)
            return self.observation_normalizer(s), r, d, i

    def reset(self):
        if hasattr(self.env, 'num_players'):
            return [self.observation_normalizer(s) for s in self.env.reset()]
        else:
            return self.observation_normalizer(self.env.reset())

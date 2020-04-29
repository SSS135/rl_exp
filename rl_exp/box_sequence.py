import numpy.random as rng
from gym.spaces import Box


class BoxSequence(Box):
    def sample(self, seq_len=None):
        assert seq_len is not None
        return rng.uniform(low=self.low, high=self.high, size=(seq_len, *self.low.shape))

    def contains(self, x):
        return x.shape[1:] == self.shape and (x >= self.low).all() and (x <= self.high).all()
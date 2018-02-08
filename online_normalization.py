import numpy as np


class OnlineNormalizer:
    def __init__(self, eps=0.01, absmax=5, scale=True, center=True):
        self.eps = eps
        self.absmax = absmax
        self.scale = scale
        self.center = center
        self.n = 0
        self._mean = 0.0
        self.M2 = 0.0
        self.std = 1

    def __call__(self, x):
        self.n += 1
        delta = x - self._mean
        self._mean = self._mean + delta / self.n if self.center else 0
        delta2 = x - self._mean
        self.M2 += delta*delta2

        var = self.M2 / self.n if self.n >= 2 else 1
        self.std = np.sqrt(np.maximum(var, self.eps)) if self.scale else 1
        self.mean = self._mean if self.n >= 2 else 0
        x = x - self.mean if self.center else x
        x = x / self.std if self.scale else x
        return np.clip(x, -self.absmax, self.absmax)

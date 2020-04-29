import math

import numpy.random as rng


class OU:
    # Ornstein–Uhlenbeck process
    def __init__(self, mu, th, sig, dt=1):
        self.th = th
        self.sig = sig
        self.dt = dt
        self.mu = mu
        self.x = mu

    def step(self):
        m = self.th * (self.mu - self.x) * self.dt
        w = self.sig * math.sqrt(self.dt) * rng.randn()
        self.x += m + w
        return self.x

    def reset(self):
        self.x = self.mu
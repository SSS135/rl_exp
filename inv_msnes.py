import time
from collections import deque

import numpy as np
import numpy.random as rng


class InvMSNES:
    def __init__(self,
                 init_mean: np.ndarray=None,
                 init_std: np.ndarray or float=0.5,
                 pop_size: int=None,
                 std_step: float=None,
                 lr_step: float=0.01,
                 lr_sample: float=0.3,
                 log_time_interval: float=None):
        self.cur_solution = init_mean
        self.lr_step = lr_step
        self.lr_sample = lr_sample
        self.std = init_std
        self.log_time_interval = log_time_interval
        self.dim = init_mean.size
        self.std_step = (3 + np.log(self.dim)) / (20 * np.sqrt(self.dim)) if std_step is None else std_step
        self.pop_size = int(round(4 + 3 * np.log(self.dim))) if pop_size is None else pop_size
        self._grad_mean = np.zeros_like(self.cur_solution)
        self.max_R = None
        self.best_solution = self.cur_solution
        self.epoch = 0
        self._log_as_best = True
        self._last_log_time = time.time()
        self._samples = None
        self._snd_sample_index = 0
        self._rcv_sample_index = 0
        self._R = deque(maxlen=pop_size)
        self._sort_idx = None
        self._noise = deque(maxlen=pop_size)
        self._new_noise = None
        self._step()

    def get_single_sample(self):
        assert self._snd_sample_index == self._rcv_sample_index
        sample = self._samples[self._snd_sample_index]
        self._snd_sample_index += 1
        return sample

    def get_batch_samples(self):
        assert self._snd_sample_index == 0
        self._snd_sample_index = len(self._samples)
        return self._samples

    def rate_single_sample(self, fitness):
        assert self._snd_sample_index == self._rcv_sample_index + 1
        self._R.append(fitness)
        self._noise.append(self._new_noise[self._rcv_sample_index])
        self._rcv_sample_index += 1

        if self._rcv_sample_index == len(self._samples):
            self._step()

    def rate_batch_samples(self, fitness):
        fitness = np.array(fitness, dtype=self._R.dtype)
        assert self._snd_sample_index == len(fitness) == len(self._samples) and self._rcv_sample_index == 0
        for f, n in zip(fitness, self._new_noise):
            self._R.append(f)
            self._noise.append(n)
        self._rcv_sample_index = len(self._samples)
        self._step()

    def _step(self):
        if len(self._R) > 1:
            self._sort_idx = np.argsort(-np.asarray(self._R))
            self._log_progress()
            self._grad_mean, grad_std = self._get_grads()
            self.cur_solution += self.lr_step * self._grad_mean
            self.std = self._get_updated_std(grad_std)
            self.epoch += 1
        self._new_noise = self._get_noise()
        self._samples = self._get_samples()
        self._snd_sample_index = self._rcv_sample_index = 0

    def _log_progress(self):
        do_log = self.log_time_interval is not None and self._last_log_time + self.log_time_interval < time.time()
        if self.max_R is None or self._R[self._sort_idx[0]] > self.max_R:
            self.max_R = self._R[self._sort_idx[0]]
            # self.best_solution = self._samples[self._sort_idx[0]]
            self._log_as_best = True
        if do_log:
            print((self.epoch + 1) * 2,
                  'new best' if self._log_as_best else 'cur iter best',
                  self.max_R if self._log_as_best else np.max(self._R),
                  'average', np.mean(self._R))
            self._log_as_best = False
            self._last_log_time = time.time()

    def _get_updated_std(self, grad_std):
        std = self.std * np.exp(self.std_step / 2 * grad_std)
        return std

    def _get_grads(self):
        noise = np.asarray(self._noise)[self._sort_idx]
        R_rank = self._get_ranks(len(self._R))
        grad_mean = np.dot(R_rank, noise)
        grad_std = np.dot(R_rank, noise ** 2 - 1)
        return grad_mean, grad_std

    def _get_samples(self):
        samples = np.empty_like(self._new_noise)
        for i in range(len(samples)):
            samples[i] = self.cur_solution + self.lr_sample * self._grad_mean + self.std * self._new_noise[i]
        return samples

    def _get_noise(self):
        noise = rng.randn(2, self.dim)
        noise[1] = -noise[0]
        return noise

    @staticmethod
    def _get_ranks(n):
        indexes = np.arange(1, n + 1)
        ranks = np.maximum(0, np.log(n / 2 + 1) - np.log(indexes))
        return ranks / ranks.sum() - 1 / n
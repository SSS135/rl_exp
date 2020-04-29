import time

import numpy as np
import numpy.random as rng


class InvSNES:
    def __init__(self,
                 init_mean: np.ndarray=None,
                 init_std: np.ndarray or float=0.5,
                 pop_size: int=None,
                 std_step: float=None,
                 lr: float=0.3,
                 log_time_interval: float=None):
        self.cur_solution = init_mean
        self.lr = lr
        self.std = init_std
        self.log_time_interval = log_time_interval
        self.dim = init_mean.size
        self.std_step = (3 + np.log(self.dim)) / (20 * np.sqrt(self.dim)) if std_step is None else std_step
        self.pop_size = int(round(4 + 3 * np.log(self.dim))) if pop_size is None else pop_size
        self.pop_size += self.pop_size % 2
        self.grad_mean = np.zeros_like(self.cur_solution)
        self.max_R = None
        self.best_solution = self.cur_solution
        self.epoch = 0
        self._R_rank = self._get_ranks(self.pop_size)
        self._log_as_best = True
        self._last_log_time = time.time()
        self._samples = None
        self._snd_sample_index = 0
        self._rcv_sample_index = 0
        self._R = np.zeros(pop_size)
        self._sort_idx = None
        self._noise = None
        self._step()

    def _step(self):
        if self._samples is not None:
            assert self._snd_sample_index == self._rcv_sample_index == len(self._samples)

            self._R = np.array(list(self._R))
            self._sort_idx = np.argsort(-self._R)

            self._log_progress()
            self.grad_mean, grad_std = self._get_grads(self._noise, self._sort_idx)
            self.cur_solution += self.lr * self.grad_mean
            self.std = self._get_updated_std(grad_std)
            self.epoch += 1

        self._noise = self._get_noise()
        self._samples = self._get_samples(self._noise, self.grad_mean)
        self._snd_sample_index = self._rcv_sample_index = 0

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
        self._R[self._rcv_sample_index] = fitness
        self._rcv_sample_index += 1

        if self._rcv_sample_index == len(self._samples):
            self._step()

    def rate_batch_samples(self, fitness):
        fitness = np.array(fitness, dtype=self._R.dtype)
        assert self._snd_sample_index == len(fitness) == len(self._samples) and self._rcv_sample_index == 0
        self._R = fitness
        self._rcv_sample_index = len(self._samples)
        self._step()

    def _log_progress(self):
        do_log = self.log_time_interval is not None and self._last_log_time + self.log_time_interval < time.time()
        if self.max_R is None or self._R[self._sort_idx[0]] > self.max_R:
            self.max_R = self._R[self._sort_idx[0]]
            self.best_solution = self._samples[self._sort_idx[0]]
            self._log_as_best = True
        if do_log:
            print((self.epoch + 1) * self.pop_size,
                  'new best' if self._log_as_best else 'cur iter best',
                  self.max_R if self._log_as_best else self._R.max(),
                  'average', self._R.mean())
            self._log_as_best = False
            self._last_log_time = time.time()

    def _get_updated_std(self, grad_std):
        std = self.std * np.exp(self.std_step / 2 * grad_std)
        return std

    def _get_grads(self, noise, sort_idx):
        noise = noise[sort_idx]
        grad_mean = np.dot(self._R_rank, noise)
        grad_std = np.dot(self._R_rank, noise ** 2 - 1)
        return grad_mean, grad_std

    def _get_noise(self):
        noise = rng.randn(self.pop_size, self.dim)
        noise[self.pop_size // 2:] = -noise[:self.pop_size // 2]
        return noise

    def _get_samples(self, noise, grad_mean):
        samples = np.empty_like(noise)
        for i in range(self.pop_size):
            samples[i] = self.cur_solution + self.lr * grad_mean + self.std * noise[i]
        return samples

    @staticmethod
    def _get_ranks(n):
        lin = np.exp(np.linspace(1, -1, n) * 3)
        lin = lin - np.mean(lin)
        lin = lin / np.std(lin) * 0.1
        return lin
        # indexes = np.arange(1, n + 1)
        # ranks = np.maximum(0, np.log(n / 2 + 1) - np.log(indexes))
        # return ranks / ranks.sum() - 1 / n
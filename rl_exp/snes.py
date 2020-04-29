import os
import time
from typing import Callable

import numpy as np
import numpy.random as rng
import torch.multiprocessing as mp

_process_fitness_callable: Callable = None


class SNES:
    def __init__(self,
                 fitness_fn: Callable,
                 init_mean: np.ndarray=None,
                 init_std: np.ndarray or float=0.5,
                 pop_size: int=None,
                 std_step: float=None,
                 lr: float=0.3,
                 log_time_interval: float=None,
                 num_processes: int=None):
        if num_processes is None:
            num_processes = os.cpu_count()

        self.fitness_fn = fitness_fn
        self.cur_solution = init_mean
        self.lr = lr
        self.std = init_std
        self.log_time_interval = log_time_interval
        self.num_processes = num_processes
        self.dim = init_mean.size
        self.std_step = (3 + np.log(self.dim)) / (20 * np.sqrt(self.dim)) if std_step is None else std_step
        self.pop_size = int(round(4 + 3 * np.log(self.dim))) if pop_size is None else pop_size
        self.pop_size += self.pop_size % 2
        self.R_rank = self.get_ranks(self.pop_size)
        self.grad_mean = np.zeros_like(self.cur_solution)
        self.max_R = None
        self.best_solution = self.cur_solution
        self._log_as_best = True
        self._last_log_time = time.time()
        self._pool = mp.Pool(initializer=self.process_fitness_init, initargs=(self.fitness_fn,)) \
            if num_processes > 1 else None

    def learn(self, fitness_evals):
        iters = int(max(1, np.ceil(fitness_evals / self.pop_size)))
        for iter in range(iters):
            noise = self.get_noise()
            samples = self.get_samples(noise, self.grad_mean)
            R, sort_idx = self.evaluate(samples)
            self.log_progress(iter, R, sort_idx, samples, iters)
            self.grad_mean, grad_std = self.get_grads(noise, sort_idx)
            self.cur_solution += self.lr * self.grad_mean
            self.std = self.get_updated_std(grad_std)
        return self.best_solution, self.max_R

    def log_progress(self, iter, R, sort_idx, samples, iter_count):
        do_log = self.log_time_interval is not None and \
                 (self._last_log_time + self.log_time_interval < time.time() or iter + 1 == iter_count)
        if self.max_R is None or R[sort_idx[0]] > self.max_R:
            self.max_R = R[sort_idx[0]]
            self.best_solution = samples[sort_idx[0]]
            self._log_as_best = True
        if do_log:
            print((iter + 1) * self.pop_size,
                  'new best' if self._log_as_best else 'cur iter best',
                  self.max_R if self._log_as_best else R.max(),
                  'average', R.mean())
            self._log_as_best = False
            self._last_log_time = time.time()

    def get_updated_std(self, grad_std):
        std = self.std * np.exp(self.std_step / 2 * grad_std)
        return std

    def get_grads(self, noise, sort_idx):
        noise = noise[sort_idx]
        grad_mean = np.dot(self.R_rank, noise)
        grad_std = np.dot(self.R_rank, noise ** 2 - 1)
        return grad_mean, grad_std

    def get_noise(self):
        noise = rng.randn(self.pop_size, self.dim)
        noise[self.pop_size // 2:] = -noise[:self.pop_size // 2]
        return noise

    def get_samples(self, noise, grad_mean):
        samples = np.empty_like(noise)
        for i in range(self.pop_size):
            samples[i] = self.cur_solution + self.lr * grad_mean + self.std * noise[i]
        return samples

    def evaluate(self, samples):
        R = self._pool.map(self.process_fitness_eval, samples) \
            if self._pool is not None else map(self.fitness_fn, samples)
        R = np.array(list(R))
        sort_idx = (-R).argsort()
        return R, sort_idx

    @staticmethod
    def get_ranks(n):
        indexes = np.arange(1, n + 1)
        ranks = np.maximum(0, np.log(n / 2 + 1) - np.log(indexes))
        return ranks / ranks.sum() - 1 / n

    @staticmethod
    def process_fitness_init(fn):
        global _process_fitness_callable
        _process_fitness_callable = fn

    @staticmethod
    def process_fitness_eval(sample):
        return _process_fitness_callable(sample)
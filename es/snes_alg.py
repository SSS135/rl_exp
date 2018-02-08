import pprint
from functools import partial

import numpy as np
import torch
from mylib.rl.common import InvSNES, RLBase
from rl.models.models import MLPActor
from torch import nn as nn
from torch.autograd import Variable


class ES(RLBase):
    def __init__(self, observation_space, action_space,
                 lr=0.1,
                 init_std_scale=1.0,
                 std_step=0.3,
                 pop_size=20,
                 model_factory=partial(MLPActor, activation=nn.ELU),
                 num_processes=None,
                 log_time_interval=None,
                 cuda=False):
        super().__init__(observation_space, action_space)
        self._init_args = locals()
        self.lr = lr
        self.init_std_scale = init_std_scale
        self.std_step = std_step
        self.pop_size = pop_size
        self.model_factory = model_factory
        self.num_processes = num_processes
        self.log_time_interval = log_time_interval
        self.cuda = cuda
        self.model = None
        self._rewards = []

        self._create_model()
        self._tensor_type = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        init_mean = self._extract_solution(self.model)
        init_std = self._extract_std(self.model)
        self.snes = InvSNES(init_mean=init_mean, init_std=init_std, pop_size=self.pop_size,
                            std_step=self.std_step, lr=self.lr, log_time_interval=None)
        self._next_model()

    @property
    def num_actors(self):
        return 1

    def _step(self, prev_states, rewards, dones, cur_states) -> np.ndarray:
        states = self._from_numpy(cur_states, dtype=np.float32)
        probs, values = self.model.forward(Variable(states, volatile=True))
        actions = self.model.pd.sample(probs.data).cpu().numpy()

        if rewards is not None:
            self._rewards.append(np.asscalar(rewards))
            if np.asscalar(dones):
                self.snes.rate_single_sample(np.sum(self._rewards))
                self._rewards.clear()
                self._next_model()

        return actions

    def _from_numpy(self, x, dtype):
        x = np.asarray(x, dtype=dtype)
        x = torch.from_numpy(x)
        if self.cuda:
            x = x.cuda()
        return x

    @staticmethod
    def _extract_solution(model: nn.Module):
        return torch.cat([p.data.cpu().view(-1) for p in model.parameters()]).numpy()

    @staticmethod
    def _fill_solution(solution, model: nn.Module):
        data_offset = 0
        for p in model.parameters():
            numel = p.numel()
            chunk = solution[data_offset: data_offset + numel]
            assert len(chunk) == numel
            chunk = torch.from_numpy(chunk).view_as(p.data).type_as(p.data)
            p.data.copy_(chunk)
            data_offset += numel

    def _create_model(self):
        self.model = self.model_factory(self.observation_space, self.action_space)
        for p in self.model.parameters():
            p.volatile = True
        if self.cuda:
            self.model = self.model.cuda()

    def _next_model(self):
        if self.model is None:
            self._create_model()
        solution = self.snes.get_single_sample()
        self._fill_solution(solution, self.model)

    def _log_set(self):
        self.model.logger = self.logger
        self.logger.add_text('PPO', pprint.pformat(self._init_args))

    @staticmethod
    def _extract_std(model: nn.Module):
        return torch.cat([p.std().cpu().expand(p.data.numel()).data for p in model.parameters()]).numpy()
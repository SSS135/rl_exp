import pprint
from functools import partial

import torch
from ppo_pytorch.common import RLBase
from rl_exp.inv_snes import InvSNES
from ppo_pytorch.actors import create_ppo_fc_actor
from torch import nn as nn


class SNES(RLBase):
    def __init__(self, observation_space, action_space,
                 lr=0.1,
                 initial_noise_scale=0.03,
                 std_step=0.3,
                 pop_size=20,
                 num_actors=1,
                 model_factory=create_ppo_fc_actor,
                 cuda=False,
                 **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self._init_args = locals()
        self.lr = lr
        self.initial_noise_scale = initial_noise_scale
        self.std_step = std_step
        self.pop_size = pop_size
        self.model_factory = model_factory
        self.cuda = cuda
        self.num_actors = num_actors
        self.model: nn.Module = None
        self._rewards = []

        assert num_actors == 1

        self._create_model()
        init_mean = self._extract_solution(self.model)
        self.snes = InvSNES(init_mean=init_mean.numpy(), init_std=initial_noise_scale, pop_size=self.pop_size,
                            std_step=self.std_step, lr=self.lr, log_time_interval=None)
        self._next_model()

    def _step(self, rewards: torch.Tensor, dones: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            ac_out = self.model(states)
            actions = self.model.heads.logits.pd.sample(ac_out.logits.data).cpu()

            if rewards is not None:
                self._rewards.append(rewards.item())
                if dones.item():
                    self.snes.rate_single_sample(sum(self._rewards))
                    self._rewards.clear()
                    self._next_model()

            return actions

    @staticmethod
    def _extract_solution(model: nn.Module):
        return torch.cat([p.data.cpu().view(-1) for p in model.parameters()])

    @staticmethod
    def _fill_solution(solution, model: nn.Module):
        data_offset = 0
        for p in model.parameters():
            numel = p.numel()
            chunk = solution[data_offset: data_offset + numel]
            assert len(chunk) == numel
            chunk = chunk.view_as(p.data).type_as(p.data)
            p.data.copy_(chunk)
            data_offset += numel

    def _create_model(self):
        self.model = self.model_factory(self.observation_space, self.action_space)
        if self.cuda:
            self.model = self.model.cuda()

    def _next_model(self):
        if self.model is None:
            self._create_model()
        solution = torch.from_numpy(self.snes.get_single_sample())
        self._fill_solution(solution, self.model)

    def _log_set(self):
        self.model.logger = self.logger
        self.logger.add_text('PPO', pprint.pformat(self._init_args))
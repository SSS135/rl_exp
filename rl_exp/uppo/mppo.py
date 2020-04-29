from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import gym.spaces as spaces
import random
from torch.utils.data import DataLoader

from .uppo import UPPO
from .uppo import ReplayBuffer
from ..models.actors import DynamicsModel
from ..models.utils import image_to_float


class MPPO(UPPO):
    """Model-based PPO (PPO + GAN dynamics model)"""

    def __init__(self, *args,
                 dynamics_batch_size=64,
                 dynamics_batches=32,
                 **kwargs):
        super().__init__(*args, **kwargs,
                         reconstruction_batch_size=dynamics_batch_size, reconstruction_batches=dynamics_batches)
        self.dynamics_batch_size = dynamics_batch_size
        self.dynamics_batches = dynamics_batches
        self.state_buffer = ReplayBuffer(100_000)

        # h_os = spaces.Box(np.array([-1] * 64), np.array([1] * 64))
        self.dyn_model = DynamicsModel(64, self.action_space, hidden_size=256)
        optim_fac = partial(optim.RMSprop, lr=1e-4)#, betas=(0.0, 0.99))
        self.dyn_optim_d = optim_fac(self.dyn_model.params_disc())
        self.dyn_optim_g = optim_fac(self.dyn_model.params_gen())

        self.dyn_dataloader = DataLoader(self.state_buffer, self.dynamics_batch_size, shuffle=True,
                                         num_workers=2, pin_memory=self.cuda_train, drop_last=True)

    # def _choose_action(self, states):
    #     return self.model(Variable(states, volatile=True))

    def _train(self):
        data = self._prepare_training_data()
        self._update_replay_buffer(data)
        self._ppo_update(data)
        self._dynamics_update()

    def _update_replay_buffer(self, data):
        def split_actors(x):
            return x.view(-1, self.num_actors, *x.shape[1:])

        def merge_actors(x):
            return x.contiguous().view(-1, *x.shape[2:])

        def to_numpy(x):
            return [np.ascontiguousarray(s) for s in x.cpu().numpy()]

        states, actions, rewards, dones = [split_actors(x) for x in
                                           (data.states, data.actions, data.rewards, data.dones)]
        next_states = states[1:]
        states, actions = [x[:-1] for x in (states, actions)]
        data = [to_numpy(merge_actors(x)) for x in (states, next_states, actions, rewards, dones)]
        states, next_states, actions, rewards, dones = data

        # for d in dones:
        #     d[d < 0.5] += random.uniform(0.01, 0.1)
        #     d[d > 0.5] -= random.uniform(0.01, 0.1)
        # for r in rewards:
        #     r += random.uniform(-0.02, 0.02)

        self.state_buffer.push(states, next_states, actions, rewards, dones)

    def _dynamics_update(self):
        batch_size = self.dynamics_batch_size
        batches = min(self.dynamics_batches, len(self.state_buffer) // batch_size)

        if len(self.state_buffer) < batch_size:
            return

        self.dyn_model.train()
        if self.cuda_train:
            self.dyn_model.cuda()

        one = Variable(torch.ones(batch_size))
        one = one.cuda() if self.cuda_train else one
        zero = Variable(one.data.clone().fill_(0))

        for i, batch in enumerate(self.dyn_dataloader):
            if i >= batches:
                break

            do_log = self._do_log and i + 1 == batches

            if self.cuda_train:
                batch = [x.cuda() for x in batch]
            state, next_state, action, reward, done = batch

            state, next_state = [Variable(image_to_float(x)) for x in (state, next_state)]
            action, reward, done = [Variable(x) for x in (action, reward, done)]

            self._dynamics_update_step(state, next_state, action, reward, done, one, zero, do_log)
            self._reconstruction_update_step(state, one, zero, do_log)

    def _dynamics_update_step(self, state, next_state, action, reward, done, one, zero, do_log):
        self.dyn_optim_d.zero_grad()
        self.dyn_optim_g.zero_grad()

        all_states = Variable(torch.cat([state.data, next_state.data], 0), volatile=True)
        state, next_state = self.model(all_states).hidden_code.data.chunk(2, 0)
        state, next_state = Variable(state), Variable(next_state)

        # real discriminator
        state = state.view(state.shape[0], -1)
        next_state = next_state.view_as(state) * (1 - done.view(-1, 1))
        real_d, hidden_real = self.dyn_model.discriminate(state, next_state, action, reward, done)
        loss_real_d = 0
        loss_real_d += F.binary_cross_entropy_with_logits(real_d, one)
        loss_real_d += 0.5 * (1 - real_d.clamp(max=1)).pow_(2).mean()
        # loss_real_d += 0.5 * (1 - real_d).pow_(2).mean()
        loss_real_d.backward()

        # fake discriminator
        fake_next_state, fake_reward, fake_done = self.dyn_model.generate(state, action)
        fake_next_state = fake_next_state.contiguous().view_as(state) * (1 - done.view(-1, 1))
        for x in (fake_next_state, fake_reward, fake_done):
            x.detach_()
        fake_d, hidden = self.dyn_model.discriminate(state, fake_next_state, action, fake_reward, fake_done)
        loss_fake_d = 0
        loss_fake_d += F.binary_cross_entropy_with_logits(fake_d, zero)
        loss_fake_d += 0.5 * (-1 - fake_d.clamp(min=-1)).pow_(2).mean()
        # loss_fake_d += 0.5 * (-1 - fake_d).pow_(2).mean()
        loss_fake_d.backward()

        self.dyn_optim_d.step()

        self.dyn_optim_d.zero_grad()
        self.dyn_optim_g.zero_grad()

        # generator
        fake_next_state, fake_reward, fake_done = self.dyn_model.generate(state, action)
        fake_next_state = fake_next_state.contiguous().view_as(state) * (1 - done.view(-1, 1))
        fake_d, hidden = self.dyn_model.discriminate(state, fake_next_state, action, fake_reward, fake_done)
        fake_head = self.model.forward_on_hidden_code(fake_next_state)
        real_head = self.model.forward_on_hidden_code(next_state)
        logp_fake = self.model.pd.logp(action, fake_head.probs)
        logp_real = self.model.pd.logp(action, real_head.probs)
        ratio = (logp_fake - logp_real).exp()
        loss_g = 0
        loss_g += F.binary_cross_entropy_with_logits(fake_d, one)
        loss_g += 0.5 * (1 - fake_d.clamp(max=1)).pow_(2).mean()
        loss_g += 0.5 * F.mse_loss(fake_head.probs, real_head.probs.detach())
        loss_g += F.mse_loss(fake_head.state_values, real_head.state_values.detach())
        # loss_g += F.mse_loss(fake_next_state, next_state)
        # loss_g += 0.33 * F.mse_loss(fake_reward, reward)
        # loss_g += 0.33 * F.mse_loss(fake_done, done)
        # loss_g += 0.5 * (1 - fake_d).pow_(2).mean()
        loss_g += F.mse_loss(hidden, hidden_real.detach())
        # loss_g += (1 - ratio).pow_(2).mean()
        loss_g /= 2
        loss_g.backward()

        self.dyn_optim_g.step()

        if do_log:
            reward_mae = (fake_reward - reward).abs_().mean()
            next_state_mae = (fake_next_state - next_state).abs_().mean()
            state_diff = (state - next_state).abs_().mean()
            done_mae = (fake_done - done).abs_().mean()
            self.logger.add_scalar('dyn reward mae', reward_mae, self.frame)
            self.logger.add_scalar('dyn next state mae', next_state_mae, self.frame)
            self.logger.add_scalar('dyn state diff mae', state_diff, self.frame)
            self.logger.add_scalar('dyn done mae', done_mae, self.frame)

            value_mae = (real_head.state_values - fake_head.state_values).abs_().mean()
            ratio = (torch.max(logp_fake, logp_real) - torch.min(logp_fake, logp_real)).exp().mean()

            self.logger.add_scalar('dyn head value mae', value_mae, self.frame)
            self.logger.add_scalar('dyn head ratio', ratio, self.frame)

            # self.logger.add_scalar('dyn state loss', state_loss, self.frame)
            # self.logger.add_scalar('dyn reward loss', reward_loss, self.frame)
            # self.logger.add_scalar('dyn done loss', done_loss, self.frame)

            # self.logger.add_scalar('dyn real d loss', loss_real_d, self.frame)
            # self.logger.add_scalar('dyn fake d loss', loss_fake_d, self.frame)
            self.logger.add_scalar('dyn gen loss', loss_g, self.frame)
            # self.logger.add_scalar('dyn real d logloss', loss_real_d.log(), self.frame)
            # self.logger.add_scalar('dyn fake d logloss', loss_fake_d.log(), self.frame)
            self.logger.add_scalar('dyn gen logloss', loss_g.log(), self.frame)

import random
from functools import partial
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from ..models import CNNActorCritic

from .ppo import PPO
from ..models.utils import image_to_float


class ReplayBuffer(Dataset):
    """
    Simple replay buffer for DQN.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        samples = list(zip(*args))
        for sample in samples:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = sample
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index):
        return self.memory[index]


class UPPO(PPO):
    """Unsupervised PPO (PPO + GAN observation reconstruction)"""

    def __init__(self, *args,
                 reconstruction_batch_size=64,
                 reconstruction_batches=32,
                 model_factory=CNNActorCritic,
                 **kwargs):
        super().__init__(*args, model_factory=model_factory, **kwargs)
        self.reconstruction_batch_size = reconstruction_batch_size
        self.reconstruction_batches = reconstruction_batches
        self.state_buffer = ReplayBuffer(50_000)

        optim_fac = partial(optim.RMSprop, lr=1e-4)  # , betas=(0.5, 0.999))
        self.rec_optim_d = optim_fac(self.model.params_disc())
        self.rec_optim_g = optim_fac(self.model.params_gen())

    def _train(self):
        data = self._prepare_training_data()
        self._reconstruction_update(data.states, data.actions)
        self._ppo_update(data)

    def _reconstruction_update(self, new_states, new_actions):
        batch_size = self.reconstruction_batch_size
        batches = self.reconstruction_batches

        new_states = new_states.cpu().numpy()
        new_states = [np.ascontiguousarray(s) for s in new_states]
        new_actions = new_actions.cpu().numpy()
        new_actions = [np.ascontiguousarray(s) for s in new_actions]
        self.state_buffer.push(new_states, new_actions)

        if len(self.state_buffer) < batch_size * batches:
            return

        all_states, all_actions = zip(*self.state_buffer.sample(batch_size * batches))
        all_states = self._from_numpy(all_states, np.uint8 if self.image_observation else np.float32,
                                      cuda=self.cuda_train)
        # all_actions = self._from_numpy(all_actions, np.int64, cuda=self.cuda_train)

        one = Variable(torch.ones(batch_size))
        one = one.cuda() if self.cuda_train else one
        zero = Variable(one.data.clone().fill_(0))

        for i in range(batches):
            do_log = self._do_log and i == 0

            slc = slice(i * batch_size, (i + 1) * batch_size)
            states = all_states[slc]
            # actions = all_actions[slc]

            states = Variable(image_to_float(states))
            # states, actions = [Variable(x) for x in (states, actions)]

            self._reconstruction_update_step(states, one, zero, do_log)

    def _reconstruction_update_step(self, states, one, zero, do_log):
        self.model.set_log(self.logger, do_log, self.step)

        realD, hidden, conv_out = self.model.discriminate(states, detach_hidden_from_conv=True)
        errD_real = 0
        errD_real += F.binary_cross_entropy_with_logits(realD, one)
        errD_real += 0.5 * (1 - realD.clamp(max=1)).pow_(2).mean()
        errD_real.backward()

        fake_state = self.model.generate(hidden, states)
        fakeD, hidden, conv_out = self.model.discriminate(fake_state.detach())
        errD_fake = 0
        errD_fake += F.binary_cross_entropy_with_logits(fakeD, zero)
        errD_fake += 0.5 * (-1 - fakeD.clamp(min=-1)).pow_(2).mean()
        errD_fake.backward()

        self.rec_optim_d.step()
        self.rec_optim_d.zero_grad()
        self.rec_optim_g.zero_grad()

        realD, hidden, conv_out_real = self.model.discriminate(states, detach_hidden_from_conv=True)
        fake_state = self.model.generate(hidden, states)
        fakeD, hidden, conv_out = self.model.discriminate(fake_state)
        errG = 0
        errG += F.binary_cross_entropy_with_logits(fakeD, one)
        errG += 0.5 * (1 - fakeD.clamp(max=1)).pow_(2).mean()
        # errG += F.mse_loss(conv_out, conv_out_real.detach())
        errG.backward()

        self.rec_optim_g.step()
        self.rec_optim_g.zero_grad()

        if do_log:
            self.logger.add_scalar('rec real d loss', errD_real, self.frame)
            self.logger.add_scalar('rec fake d loss', errD_fake, self.frame)
            self.logger.add_scalar('rec gen loss', errG, self.frame)

            self.logger.add_scalar('rec real d logloss', errD_real.log(), self.frame)
            self.logger.add_scalar('rec fake d logloss', errD_fake.log(), self.frame)
            self.logger.add_scalar('rec gen logloss', errG.log(), self.frame)

        self.model.set_log(self.logger, False, self.step)
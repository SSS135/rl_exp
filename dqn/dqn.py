import copy
import time

import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from ppo_pytorch.common import RLBase
from ppo_pytorch.common import ValueDecay
from ppo_pytorch.models import MLPActionValues
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
from torchvision.utils import make_grid

from .replay_buffer import ReplayBuffer


class DQN(RLBase):
    def __init__(self, observation_space, action_space,
                 optim=optim.Adam,
                 eval_batch_size=1,
                 replay_batch_size=32,
                 trainable_batch_size=64,
                 train_interval=4,
                 log_time_interval=5,
                 replay_size=50_000,
                 target_net_update_freq=1000,
                 eps=ValueDecay(1, 0, 1e5, exp=False),
                 reward_discount=0.995,
                 model_factory=MLPActionValues,
                 loss_fn=nn.SmoothL1Loss(reduce=False),
                 cuda_eval=False,
                 cuda_replay=False,
                 double=True,
                 learning_starts=2048,
                 reward_scale=0.2,
                 lr_scheduler=None,
                 max_grad_norm=0.5):
        super().__init__(observation_space, action_space)
        assert isinstance(action_space, gym.spaces.Discrete)

        self.eval_batch_size = eval_batch_size
        self.replay_batch_size = replay_batch_size
        self.trainable_batch_size = trainable_batch_size
        self.log_time_interval = log_time_interval
        self.replay_size = replay_size
        self.target_net_update_freq = target_net_update_freq
        self.eps = eps
        self.reward_discount = reward_discount
        self.cuda_eval = cuda_eval
        self.cuda_replay = cuda_replay
        self.double = double
        self.learning_starts = learning_starts
        self.step = 0
        self.reward_scale = reward_scale
        self.train_interval = train_interval
        self.loss_fn = loss_fn
        self.max_grad_norm = max_grad_norm

        self.target_network_update_step = 0
        self.processed_steps = 0
        self.actions = None
        self.LongTensor = torch.cuda.LongTensor if cuda_replay else torch.LongTensor
        self.Tensor = torch.cuda.FloatTensor if cuda_replay else torch.FloatTensor

        self.replay_buffer = ReplayBuffer(capacity=replay_size)
        self.net = model_factory(observation_space, action_space)
        self.optim = optim(self.net.parameters())
        self.target_net = None
        self.lr_scheduler = lr_scheduler(self.optim) if lr_scheduler is not None else None

        if not hasattr(self.eps, 'value'):
            self.eps = ValueDecay(self.eps, self.eps, 1)

    @property
    def num_actors(self):
        return self.eval_batch_size

    def _step(self, rewards, dones, states):
        self._check_log()

        states = Variable(torch.from_numpy(states), volatile=True)
        if self.cuda_eval:
            states = states.cuda()
            self.net = self.net.cuda()
        else:
            self.net = self.net.cpu()

        self.net.eval()
        ac_out = self.net(states)
        q_values = ac_out.action_values

        if len(self.replay_buffer) < self.learning_starts:
            q_values.data.fill_(0.5).bernoulli_()
        _, actions = q_values.max(dim=1)
        actions = np.asarray(actions.data.cpu().long().view(-1).numpy())

        prev_actions = self.actions
        self.actions = actions

        if prev_states is not None:
            replay_next_states = [(None if d else s) for s, d in zip(states, dones)]
            self.replay_buffer.push(prev_states, prev_actions, replay_next_states, rewards)

        if self.step % self.train_interval == 0 and \
                        len(self.replay_buffer) >= max(self.replay_batch_size, self.learning_starts):
            self.learn_q()

        self.step += 1
        self.eps.step()

        return self.actions

    def from_numpy(self, x, dtype):
        x = np.asarray(x, dtype=dtype)
        x = torch.from_numpy(x)
        if self.cuda_replay:
            x = x.cuda()
        return x

    def learn_q(self):
        # init

        self.net = self.net.cuda() if self.cuda_replay else self.net.cpu()
        self.net.train()

        if self.target_net is None or self.step > self.target_network_update_step + self.target_net_update_freq:
            self.target_network_update_step = self.step
            if self.target_net is None:
                self.target_net = copy.deepcopy(self.net)
            else:
                for pc, pt in zip(self.net.parameters(), self.target_net.parameters()):
                    pt.data.copy_(pc.data)

        samples = self.replay_buffer.sample(self.replay_batch_size)
        norm_rewards = (self.reward_scale * np.asarray(samples.rewards)).clip(-1, 1)
        next_states_available = self.from_numpy([s is not None for s in samples.next_states], dtype=np.float32)
        next_states = self.from_numpy([s if s is not None else samples.states[0]
                                       for s in samples.next_states], dtype=np.float32)
        states = self.from_numpy(samples.states, dtype=np.float32)
        actions = self.from_numpy(samples.actions, dtype=np.int64)
        rewards = self.from_numpy(norm_rewards, dtype=np.float32)

        states, actions, rewards, next_states, next_states_available = \
            [Variable(x.cuda() if self.cuda_replay else x.cpu())
             for x in (states, actions, rewards, next_states, next_states_available)]

        # training

        self.optim.zero_grad()

        self.net.allow_noise = True
        ac_out_cur_online = self.net(states)
        self.net.allow_noise = False
        ac_out_next_online = self.net(next_states)
        self.target_net.allow_noise = False
        ac_out_next_target = self.target_net(next_states)

        cur_action_value, target_action_value = \
            self.get_q_loss(ac_out_cur_online.action_values,
                            ac_out_next_online.action_values,
                            ac_out_next_target.action_values,
                            rewards, actions, next_states_available)
        av_loss = self.loss_fn(cur_action_value, target_action_value)

        loss = av_loss.sort()[0][-self.trainable_batch_size:].mean()
        loss.backward()
        clip_grad_norm(self.net.parameters(), self.max_grad_norm)
        self.optim.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # logging

        if self._do_log:
            if states.dim() == 4:
                img = states[:4]
                img = img.view(-1, *img.shape[2:]).unsqueeze(1)
                img = make_grid(img, nrow=states.shape[1], normalize=True)
                self.logger.add_image('state', img, self.step)
            self.logger.add_histogram('rewards', rewards, self.step)
            self.logger.add_histogram('td errors', cur_action_value - target_action_value, self.step)
            self.logger.add_histogram('norm rewards', norm_rewards, self.step)
            self.logger.add_scalar('cur eps', self.eps.value, self.step)
            self.logger.add_scalar('total loss', loss, self.step)
            if self.lr_scheduler is not None:
                self.logger.add_scalar('learning rate', self.lr_scheduler.get_lr()[0], self.step)
            for name, param in self.net.named_parameters():
                self.logger.add_histogram(name, param, self.step)

    def get_q_loss(self, q_cur_online, q_next_online, q_next_target, rewards, actions, next_states_avail):
        gather_idx = actions.view(-1, 1).expand(q_cur_online.size(0), 1)
        q_cur = q_cur_online.gather(1, index=gather_idx).squeeze()
        next_states_avail = next_states_avail.data.view(-1, 1)
        q_next_target = q_next_target.data * next_states_avail

        if self.double:
            q_next_online = q_next_online.data * next_states_avail
            ac_idx = q_next_online.max(1, keepdim=True)[1]
            q_next = q_next_target.gather(1, index=ac_idx).squeeze()
        else:
            q_next = q_next_target.max(1)[0]
        q_next = Variable(q_next)

        target = self.reward_discount * q_next + rewards

        if self._do_log:
            self.logger.add_histogram('action values selected', q_cur.view(-1), self.step)
            self.logger.add_histogram('action values all', q_cur_online.view(-1), self.step)
            self.logger.add_scalar('action values loss', self.loss_fn(q_cur, target).mean(), self.step)

        return q_cur, target

    def _check_log(self):
        if self.logger is not None and self.log_time_interval is not None and \
                                self._last_log_time + self.log_time_interval < time.time():
            self._last_log_time = time.time()
            self._do_log = True
        else:
            self._do_log = False

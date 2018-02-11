import copy
from functools import partial

import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler
from ppo_pytorch.common import RLBase
from ppo_pytorch.common import ValueDecay
from ppo_pytorch.common.multi_dataset import MultiDataset
from ppo_pytorch.experimental.dqn import ReplayBuffer
from ppo_pytorch.models import MLPActionValues, DynamicsModel
from sklearn.metrics import precision_recall_fscore_support
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).logger()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


def dropout_exploration(action_values, drop_p):
    action_values = action_values.data.clone()
    rand_vec = action_values.new(action_values.shape).fill_(drop_p).bernoulli_().byte()
    action_values.masked_fill_(rand_vec, -1e8)
    return Variable(action_values.max(dim=1)[1])


# def eps_greedy_exploration(action_values, eps):
#     action_values = action_values.data.clone()
#     rand_vec = action_values.new(action_values.size(0)).fill_(eps).bernoulli_().unsqueeze(1).byte()
#     action_values.masked_fill_(rand_vec, -1e8)
#     return Variable(action_values.max(-1)[1])


class MDQN(RLBase):
    def __init__(self, observation_space, action_space,
                 policy_optimizer_factory=partial(optim.Adam, lr=2e-4, weight_decay=1e-5),
                 dynamics_optimizer_factory=partial(optim.Adam, lr=5e-4, weight_decay=5e-5),
                 num_actors=1,
                 policy_batch_size=64,
                 dynamics_batch_size=512,
                 policy_used_batch_size=32,
                 dynamics_used_batch_size=256,
                 train_interval=128,
                 train_samples=2*1024,
                 dynamics_discount=0.9,
                 replay_size=100_000,
                 target_network_update_interval=1024,
                 exploration=ValueDecay(0.5, 0.02, 0.5e5),
                 reward_discount=0.995,
                 policy_model_factory=MLPActionValues,
                 dynamics_model_factory=DynamicsModel,
                 # loss_fn='huber',
                 cuda_run=False,
                 cuda_train=False,
                 double_q_learning=True,
                 policy_learning_starts=10000,
                 reward_scale=0.2,
                 lr_scheduler=None,
                 log_time_interval=5, ):
        super().__init__(observation_space, action_space, log_time_interval=log_time_interval)
        assert isinstance(action_space, gym.spaces.Discrete)
        assert train_samples % policy_batch_size == 0 and train_samples % dynamics_batch_size == 0
        assert policy_used_batch_size <= policy_batch_size and dynamics_used_batch_size <= dynamics_batch_size
        assert train_samples >= policy_batch_size and train_samples >= dynamics_batch_size
        assert replay_size >= train_samples

        self._num_actors = num_actors
        self.policy_batch_size = policy_batch_size
        self.dynamics_batch_size = dynamics_batch_size
        self.policy_used_batch_size = policy_used_batch_size
        self.dynamics_used_batch_size = dynamics_used_batch_size
        self.log_time_interval = log_time_interval
        self.replay_size = replay_size
        self.exploration = exploration
        self.reward_discount = reward_discount
        self.dynamics_discount = dynamics_discount
        self.cuda_run = cuda_run
        self.cuda_train = cuda_train
        self.double_q_learning = double_q_learning
        self.policy_learning_starts = max(policy_learning_starts, train_samples)
        self.reward_scale = reward_scale
        self.train_interval = train_interval
        self.train_samples = train_samples
        self.target_network_update_interval = target_network_update_interval
        # self.loss_fn = LOSSES[loss_fn]

        self.actions = None
        self._log_stats = dict()

        self.replay_buffer = ReplayBuffer(capacity=replay_size)
        self.policy_model = policy_model_factory(observation_space, action_space)
        self.policy_optimizer = policy_optimizer_factory(self.policy_model.parameters())
        self.dynamics_model = dynamics_model_factory(observation_space, action_space)
        self.dynamics_optimizer = dynamics_optimizer_factory(self.dynamics_model.parameters())
        self.target_policy_model = copy.deepcopy(self.policy_model)
        self.lr_scheduler = lr_scheduler(self.policy_optimizer) if lr_scheduler is not None else None

        if not hasattr(self.exploration, 'value'):
            self.exploration = ValueDecay(self.exploration, self.exploration, 1)

    @property
    def num_actors(self):
        return self._num_actors

    def _step(self, prev_states, rewards, dones, cur_states):
        states = Variable(torch.from_numpy(cur_states), volatile=True)
        if self.cuda_run:
            states = states.cuda()
            self.policy_model = self.policy_model.cuda()
        else:
            self.policy_model = self.policy_model.cpu()

        self.policy_model.eval()
        ac_out = self.policy_model(states)
        q_values = ac_out.action_values

        if len(self.replay_buffer) < self.policy_learning_starts:
            q_values.data.fill_(0.5).bernoulli_()
        actions = q_values.max(dim=1)[1] # dropout_exploration(q_values, self.exploration.value)
        actions = np.asarray(actions.data.cpu().long().view(-1).numpy())

        prev_actions = self.actions
        self.actions = actions

        if prev_states is not None:
            replay_next_states = [(None if d else s) for s, d in zip(cur_states, dones)]
            self.replay_buffer.push(prev_states, prev_actions, replay_next_states, rewards)

        if self.step % self.target_network_update_interval == 0:
            for pc, pt in zip(self.policy_model.parameters(), self.target_policy_model.parameters()):
                pt.data.copy_(pc.data)

        if self.step % self.train_interval == 0 and len(self.replay_buffer) >= self.train_samples:
            self._train()

        self.exploration.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self._do_log:
            self.logger.add_scalar('exploration', self.exploration.value, self.frame)

        return self.actions

    def _train(self):
        # init

        self._check_log()

        self.policy_model = self.policy_model.cuda() if self.cuda_train else self.policy_model.cpu()
        self.target_policy_model = self.target_policy_model.cuda() if self.cuda_train else self.target_policy_model.cpu()
        self.dynamics_model = self.dynamics_model.cuda() if self.cuda_train else self.dynamics_model.cpu()
        self.policy_model.train()
        self.target_policy_model.train()

        samples = self.replay_buffer.sample(self.train_samples)
        norm_rewards = (self.reward_scale * np.asarray(samples.rewards)).clip(-1, 1)
        next_states_available = self.from_numpy([s is not None for s in samples.next_states],
                                                dtype=np.float32)
        next_states = self.from_numpy([s if s is not None else samples.states[0]
                                       for s in samples.next_states], dtype=np.float32)
        states = self.from_numpy(samples.states, dtype=np.float32)
        actions = self.from_numpy(samples.actions, dtype=np.int64)
        rewards = self.from_numpy(norm_rewards, dtype=np.float32)

        dataset = MultiDataset(states, actions, rewards, next_states, next_states_available)
        policy_dataloader = DataLoader(
            dataset, batch_size=self.policy_batch_size, shuffle=True, pin_memory=self.cuda_train)
        dynamics_dataloader = DataLoader(
            dataset, batch_size=self.dynamics_batch_size, shuffle=True, pin_memory=self.cuda_train)

        # training

        for batch_idx, batch in enumerate(dynamics_dataloader):
            batch = [Variable(x.cuda() if self.cuda_train else x) for x in batch]
            self._train_dynamics(*batch, batch_idx)

        if len(self.replay_buffer) >= self.policy_learning_starts:
            for batch_idx, batch in enumerate(policy_dataloader):
                batch = [Variable(x.cuda() if self.cuda_train else x) for x in batch]
                self._train_policy(*batch, batch_idx)

        self._flush_log_stats()

        # logging

        if self._do_log:
            if states.dim() == 4:
                img = states[:4]
                img = img.view(-1, *img.shape[2:]).unsqueeze(1)
                img = make_grid(img, nrow=states.shape[1], normalize=True)
                self.logger.add_image('state', img, self.frame)
            self.logger.add_histogram('rewards', rewards, self.frame)
            self.logger.add_histogram('norm rewards', norm_rewards, self.frame)
            self.logger.add_scalar('cur eps', self.exploration.value, self.frame)
            if self.lr_scheduler is not None:
                self.logger.add_scalar('learning rate', self.lr_scheduler.get_lr()[0], self.frame)
            for name, param in self.policy_model.named_parameters():
                self.logger.add_histogram(name, param, self.frame)

    def _train_dynamics(self, states, actions, rewards, next_states, next_states_available, batch_idx):
        p_next_state, p_r, p_done = self.dynamics_model(states, actions)
        state_dyn_loss = F.mse_loss(p_next_state, next_states, reduce=False) * next_states_available.unsqueeze(1)
        r_dyn_loss = F.mse_loss(p_r, rewards, reduce=False)
        dones = 1 - next_states_available
        done_loss_mult = ((p_done - dones).abs() > 0.1).float().detach()
        done_dyn_loss = binary_cross_entropy_with_logits(p_done, dones, reduce=False) * done_loss_mult
        dyn_loss = state_dyn_loss.mean(-1) + r_dyn_loss + done_dyn_loss
        dyn_loss = dyn_loss.sort()[0][-self.dynamics_used_batch_size:].mean()
        dyn_loss = dyn_loss * dyn_loss
        dyn_loss.backward()
        clip_grad_norm(self.dynamics_model.parameters(), 20)
        self.dynamics_optimizer.step()
        self.dynamics_optimizer.zero_grad()

        if self._do_log:
            self._log_histogram_buffered('dynamics reward error', r_dyn_loss)
            self._log_histogram_buffered('dynamics state error', state_dyn_loss)
            self._log_histogram_buffered('dynamics done error', done_dyn_loss)
            self._log_scalar_buffered('dynamics total error', dyn_loss)

            p_done = F.sigmoid(p_done).data.round().cpu().numpy().astype(np.bool)
            dones = dones.data.cpu().numpy().astype(np.bool)
            valid_precision = p_done.sum() != 0
            valid_recall = dones.sum() != 0
            warn_for = tuple((['precision'] if valid_precision else []) + (['recall'] if valid_recall else []))
            precision, recall, _, _ = precision_recall_fscore_support(
                dones, p_done, average='binary', warn_for=warn_for)
            if valid_precision:
                self._log_scalar_buffered('dynamics done precision', precision)
            if valid_recall:
                self._log_scalar_buffered('dynamics done recall', recall)

    def _train_policy(self, states, actions, rewards, next_states, next_states_available, batch_idx):
        ac_out_cur_online = self.policy_model(states)

        # actions = eps_greedy_exploration(ac_out_cur_online.action_values, 0.5).detach()
        actions = Variable(actions.data.clone().random_(self.action_space.n))
        next_states, rewards, dones = [x.detach() for x in self.dynamics_model(states, actions)]
        dones = (F.sigmoid(dones) * 1.4 - 0.2).clamp(0, 1)
        next_states_available = 1 - dones

        ac_out_next_online = self.policy_model(next_states)
        ac_out_next_target = self.target_policy_model(next_states)

        cur_action_value, target_action_value = self._get_q_loss(
            ac_out_cur_online.action_values,
            ac_out_next_online.action_values,
            ac_out_next_target.action_values,
            'action values', rewards, actions, next_states_available)

        policy_loss = F.smooth_l1_loss(cur_action_value, target_action_value, reduce=False)
        policy_loss = policy_loss.sort()[0][-self.policy_used_batch_size:].mean()
        policy_loss.backward()
        clip_grad_norm(self.policy_model.parameters(), 20)
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()

        if self._do_log:
            self._log_histogram_buffered('td errors', cur_action_value - target_action_value)
            self._log_histogram_buffered('action values selected', cur_action_value.view(-1))
            self._log_scalar_buffered('action values loss', policy_loss)

    def _get_q_loss(self, q_cur_online, q_next_online, q_next_target, q_name, rewards, actions, next_states_avail):
        gather_idx = actions.unsqueeze(1)
        q_cur = q_cur_online.gather(1, index=gather_idx).squeeze()
        next_states_avail = next_states_avail.data.unsqueeze(1)
        q_next_target = q_next_target.data * next_states_avail

        if self.double_q_learning:
            q_next_online = q_next_online.data * next_states_avail
            ac_idx = q_next_online.max(1, keepdim=True)[1]
            q_next = q_next_target.gather(1, index=ac_idx).squeeze()
        else:
            q_next = q_next_target.max(1)[0].squeeze()
        q_next = Variable(q_next)

        target = self.reward_discount * q_next + rewards

        if self._do_log:
            self._log_histogram_buffered('{} all'.format(q_name), q_cur_online.view(-1))

        return q_cur, target

    def from_numpy(self, x, dtype):
        x = np.asarray(x, dtype=dtype)
        x = torch.from_numpy(x)
        if self.cuda_train:
            x = x.cuda()
        return x

    def _log_scalar_buffered(self, name, data):
        return self._record_for_log(name, data, False)

    def _log_histogram_buffered(self, name, data):
        return self._record_for_log(name, data, True)

    def _record_for_log(self, name, data, histogram):
        if isinstance(data, Variable):
            data = data.data
        if data.__class__.__name__.find('Tensor') != -1:
            data = data.cpu().numpy()
        key = ('___hist___' if histogram else '___scalar___') + name
        stats = self._log_stats.get(key, ([],))[0]
        self._log_stats[key] = (stats, histogram, name)
        stats.append(data)

    def _flush_log_stats(self):
        for stats, histogram, name in self._log_stats.values():
            if histogram:
                self.logger.add_histogram(name, np.asarray(stats), self.frame)
            else:
                self.logger.add_scalar(name, np.mean(stats).item(), self.frame)
        self._log_stats.clear()

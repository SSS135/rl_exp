import copy
import pprint
from asyncio import Future
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.autograd
import torch.optim as optim
from ppo_pytorch.algs.ppo import SchedulerManager, copy_state_dict, log_training_data
from torch.nn.utils import clip_grad_norm_

from ppo_pytorch.algs.replay_buffer import ReplayBuffer
from ppo_pytorch.algs.utils import RunningNorm, scaled_impala_loss
from ppo_pytorch.algs.utils import v_mpo_loss
from ppo_pytorch.actors import create_ppo_fc_actor, Actor
from ppo_pytorch.actors.activation_norm import activation_norm_loss
from ppo_pytorch.actors.utils import model_diff
from ppo_pytorch.algs.utils import lerp_module_
from ppo_pytorch.common.attr_dict import AttrDict
from ppo_pytorch.common.barron_loss import barron_loss
from ppo_pytorch.common.data_loader import DataLoader
from ppo_pytorch.common.gae import calc_vtrace, calc_value_targets
from ppo_pytorch.common.pop_art import PopArt
from ppo_pytorch.common.rl_base import RLBase
from rl_exp.inv_snes import InvSNES


def wiener(dt=0.1, x0=0, decay=0.95, n=100):
    res = np.zeros(n)
    res[0] = x0
    for i in range(1, n):
        res[i] = decay * res[i - 1] + dt * np.random.randn()
    return (res - res.mean()) / res.std()


def get_ranks(n, temp=1):
    lin = np.exp(np.linspace(-1, 1, n) * temp)
    lin = lin - np.mean(lin)
    lin = lin / np.std(lin) * 0.1
    return lin


class GradES(RLBase):
    def __init__(self, observation_space, action_space,
                 es_learning_rate=0.1,
                 models_per_update=2,
                 episodes_per_model=None,
                 horizon=64,
                 batch_size=64,
                 model_factory=create_ppo_fc_actor,
                 optimizer_factory=partial(optim.Adam, lr=3e-4),
                 entropy_loss_scale=0.01,
                 cuda_eval=False,
                 cuda_train=False,
                 barron_alpha_c=(1.5, 1),
                 lr_scheduler_factory=None,
                 clip_decay_factory=None,
                 entropy_decay_factory=None,
                 replay_buf_size=32 * 1024,
                 num_batches=8,
                 min_replay_size=10000,
                 grad_clip_norm=None,
                 kl_pull=0.1,
                 replay_end_sampling_factor=1.0,
                 eval_model_blend=0.1,
                 **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self._init_args = locals()
        self.es_learning_rate = es_learning_rate
        self.models_per_update = models_per_update
        self.episodes_per_model = self.num_actors if episodes_per_model is None else episodes_per_model
        self.entropy_loss_scale = entropy_loss_scale
        self.horizon = horizon
        self.batch_size = batch_size
        self.device_eval = torch.device('cuda' if cuda_eval else 'cpu')
        self.device_train = torch.device('cuda' if cuda_train else 'cpu')
        self.grad_clip_norm = grad_clip_norm
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.barron_alpha_c = barron_alpha_c
        self.lr_scheduler_factory = lr_scheduler_factory
        self.replay_buf_size = replay_buf_size
        self.num_batches = num_batches
        self.min_replay_size = min_replay_size
        self.replay_end_sampling_factor = replay_end_sampling_factor
        self.kl_pull = kl_pull
        self.eval_model_blend = eval_model_blend

        assert self.batch_size % self.horizon == 0, (self.batch_size, self.horizon)
        assert models_per_update % 2 == 0

        self._train_model: Actor = model_factory(observation_space, action_space).to(self.device_train).train()
        if self.model_init_path is not None:
            self._train_model.load_state_dict(torch.load(self.model_init_path), True)
            print(f'loaded model {self.model_init_path}')
        self._eval_model = model_factory(observation_space, action_space).to(self.device_eval).eval()
        self._smooth_model = model_factory(observation_space, action_space).to(self.device_train).train()
        self._central_model = model_factory(observation_space, action_space).to(self.device_train).train()
        self._eval_model.load_state_dict(self._train_model.state_dict())
        self._smooth_model.load_state_dict(self._train_model.state_dict())
        self._central_model.load_state_dict(self._train_model.state_dict())

        self._models = []
        self._models_fitness = []
        self._model_index = 0
        self._num_completed_episodes = torch.zeros(self.num_actors, dtype=torch.long)
        self._episode_returns = torch.zeros(self.num_actors)

        # self._train_future: Optional[Future] = None
        # self._executor = ThreadPoolExecutor(max_workers=1)

        self._optimizer = optimizer_factory(self._train_model.parameters())
        self._scheduler = SchedulerManager(self._optimizer, lr_scheduler_factory, clip_decay_factory, entropy_decay_factory)

        self._replay_buffer = ReplayBuffer(replay_buf_size)
        self._prev_data = None
        self._eval_steps = 0

    def _step(self, rewards: torch.Tensor, dones: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # run network
            ac_out = self._eval_model(states.to(self.device_eval))
            ac_out.state_values = ac_out.state_values.squeeze(-1)
            actions = self._eval_model.heads.logits.pd.sample(ac_out.logits).cpu()

            assert not torch.isnan(actions.sum())

            self._eval_steps += 1

            if not self.disable_training:
                if self._prev_data is not None and rewards is not None:
                    self._replay_buffer.push(rewards=rewards, dones=dones, **self._prev_data)

                    for i in range(len(dones)):
                        self._episode_returns[i] += rewards[i]
                        if dones[i] > 0:
                            if len(self._models_fitness) <= self._model_index:
                                self._models_fitness.append([])
                            if self._num_completed_episodes[i].item() >= 0:
                                self._models_fitness[self._model_index].append(self._episode_returns[i].item())
                            self._episode_returns[i] = 0

                enough_episodes = self._num_completed_episodes.sum().item() >= self.episodes_per_model and \
                                  self._num_completed_episodes.min().item() > 0
                enough_frames = len(self._replay_buffer) >= self.min_replay_size
                no_models = len(self._models) == 0
                if (enough_episodes or no_models) and enough_frames:
                    self._eval_steps = 0
                    self._es_train()
                    self._num_completed_episodes.fill_(-1)

                if self.dones is not None:
                    self._num_completed_episodes += dones.long()
                self._prev_data = AttrDict(**ac_out, states=states, actions=actions)

            return actions

    def _es_train(self):
        self.step_train = self.step_eval

        if len(self._models) <= self._model_index + 1:
            if len(self._models) > 0:
                self._update_central_model()
            self._generate_population()
            self._model_index = 0
        else:
            self._model_index += 1

        self._eval_model.load_state_dict(self._models[self._model_index])

    def _update_central_model(self):
        sort_idx = np.argsort([np.mean(f) for f in self._models_fitness])
        weights = get_ranks(len(self._models), 2)
        models = [self._models[i] for i in sort_idx]

        for name, central_tensor in self._central_model.state_dict().items():
            grad = [weight * (model[name] - central_tensor) for (model, weight) in zip(models, weights)]
            grad = torch.sum(torch.stack(grad, 0), 0)
            central_tensor += self.es_learning_rate * grad

        self._model_saver.check_save_model(self._central_model, self.frame_train)
        lerp_module_(self._smooth_model, self._central_model, self.eval_model_blend)

    def _generate_population(self):
        self._models = []
        self._models_fitness = []
        self._scheduler.step(self.frame_train)
        for _ in range(self.models_per_update // 2):
            self._generate_model_pair()

    def _generate_model_pair(self):
        data = self._create_data()
        # data.random_grad = torch.tensor(wiener(n=data.logits.numel())).view_as(data.logits)
        data.random_grad = torch.tensor(wiener(n=data.logits.numel())).view(data.logits.shape[1], data.logits.shape[0], *data.logits.shape[2:]).transpose(0, 1)
        # data.random_grad = calc_value_targets(data.rewards, torch.zeros((data.state_values.shape[0] + 1, *data.state_values.shape[1:])), data.dones, 0.99)
        # data.random_grad = (data.random_grad - data.random_grad.mean()) / (data.random_grad.std() + 0.001)
        self._check_log()
        rng_state = torch.get_rng_state()
        for positive_grad in (True, False):
            self._train_model.load_state_dict(self._central_model.state_dict())
            torch.set_rng_state(rng_state)
            # print('set state')
            # for t in self._train_model.parameters():
            #     rand = torch.randn_like(t)
            #     # print(t.pow(2).mean().sqrt())
            #     t += 0.03 * (1 if positive_grad else -1) * rand
            self._train_update(data, positive_grad)
            self._models.append(copy.deepcopy(self._train_model.state_dict()))
            self._do_log = False

    def _create_data(self):
        def cat_replay(last, rand):
            # (H, B, *) + (H, B * replay, *) = (H, B, 1, *) + (H, B, replay, *) =
            # = (H, B, replay + 1, *) = (H, B * (replay + 1), *)
            H, B, *_ = last.shape
            all = torch.cat([
                last.unsqueeze(2),
                rand.reshape(H, B, self.num_batches, *last.shape[2:])], 2)
            return all.reshape(H, B * (self.num_batches + 1), *last.shape[2:])

        h_reduce = self.horizon // self.horizon

        def fix_on_policy_horizon(v):
            return v.reshape(h_reduce, self.horizon, *v.shape[1:])\
                .transpose(0, 1)\
                .reshape(self.horizon, h_reduce * v.shape[1], *v.shape[2:])

        # (H, B, *)
        last_samples = self._replay_buffer.get_last_samples(self.horizon)
        last_samples = {k: fix_on_policy_horizon(v) for k, v in last_samples.items()}
        if self.num_batches != 0 and len(self._replay_buffer) >= \
                max(self.horizon * self.num_actors * max(1, self.num_batches), self.min_replay_size):
            num_rollouts = self.num_actors * self.num_batches * h_reduce
            rand_samples = self._replay_buffer.sample(num_rollouts, self.horizon, self.replay_end_sampling_factor)
            return AttrDict({k: cat_replay(last, rand)
                             for (k, rand), last in zip(rand_samples.items(), last_samples.values())})
        else:
            return AttrDict(last_samples)

    def _train_update(self, data: AttrDict, positive_grad: bool):
        num_samples = data.states.shape[0] * data.states.shape[1]
        num_rollouts = data.states.shape[1]

        data = AttrDict(states=data.states, logits_old=data.logits, random_grad=data.random_grad,
                        actions=data.actions, rewards=data.rewards, dones=data.dones)

        num_batches = max(1, num_samples // self.batch_size)
        rand_idx = torch.arange(num_rollouts, device=self.device_train).chunk(num_batches)
        assert len(rand_idx) == num_batches

        old_model = {k: v.clone() for k, v in self._train_model.state_dict().items()}
        kls_policy = []
        kls_replay = []

        # for t in self._train_model.parameters():
        #     t += 0.03 * torch.randn_like(t)

        with DataLoader(data, rand_idx, self.device_train, 4, dim=1) as data_loader:
            for batch_index in range(num_batches):
                batch = AttrDict(data_loader.get_next_batch())
                loss = self._train_step(batch, positive_grad, self._do_log and batch_index == num_batches - 1)
                kls_policy.append(batch.kl_smooth.mean().item())
                kls_replay.append(batch.kl_replay.mean().item())

        kl_policy = np.mean(kls_policy)
        kl_replay = np.mean(kls_replay)

        if self._do_log:
            if loss is not None:
                self.logger.add_scalar('total_loss', loss, self.frame_train)
            self.logger.add_scalar('kl', kl_policy, self.frame_train)
            self.logger.add_scalar('kl_replay', kl_replay, self.frame_train)
            self.logger.add_scalar('model_abs_diff', model_diff(old_model, self._train_model), self.frame_train)
            self.logger.add_scalar('model_max_diff', model_diff(old_model, self._train_model, True), self.frame_train)

    def _train_step(self, batch, positive_grad, do_log):
        with torch.enable_grad():
            actor_params = AttrDict()
            if do_log:
                actor_params.logger = self.logger
                actor_params.cur_step = self.frame_train

            actor_out = self._train_model(batch.states.reshape(-1, *batch.states.shape[2:]), **actor_params)
            with torch.no_grad():
                actor_out_smooth = self._smooth_model(batch.states.reshape(-1, *batch.states.shape[2:]))

            batch.logits = actor_out.logits.reshape(*batch.states.shape[:2], *actor_out.logits.shape[1:])
            batch.logits_smooth = actor_out_smooth.logits.reshape(*batch.states.shape[:2], *actor_out.logits.shape[1:])
            batch.state_values = actor_out.state_values.reshape(*batch.states.shape[:2])

            for k, v in list(batch.items()):
                batch[k] = v if k == 'states' else v.cpu()

            loss = self._get_loss(batch, positive_grad, do_log)
            #act_norm_loss = activation_norm_loss(self._train_model).cpu()
            loss = loss.mean() #+ 0.003 * act_norm_loss

        # if do_log:
        #     self.logger.add_scalar('activation_norm_loss', act_norm_loss, self.frame_train)

        # optimize
        loss.backward()
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._optimizer.step()
        self._optimizer.zero_grad()

        return loss

    def _get_loss(self, data, positive_grad, do_log=False, pd=None, tag=''):
        if pd is None:
            pd = self._train_model.heads.logits.pd

        # action probability ratio
        data.logp_old = pd.logp(data.actions, data.logits_old).sum(-1)
        data.logp_policy = pd.logp(data.actions, data.logits_smooth).sum(-1)
        data.logp = pd.logp(data.actions, data.logits).sum(-1)
        data.probs_ratio = (data.logp.detach() - data.logp_old).exp()
        data.kl_replay = pd.kl(data.logits, data.logits_old).sum(-1)
        data.kl_smooth = pd.kl(data.logits, data.logits_smooth).sum(-1)
        entropy = pd.entropy(data.logits).sum(-1)

        loss_policy = data.logits * data.random_grad * (1 if positive_grad else -1)
        loss_kl = self.kl_pull * data.kl_smooth
        loss_ent = self.entropy_loss_scale * -entropy

        loss_policy = loss_policy.mean()
        loss_kl = loss_kl.mean()
        loss_ent = loss_ent.mean()

        if do_log:
            self.logger.add_scalar('loss_alpha' + tag, loss_kl, self.frame_train)

        # sum all losses
        total_loss = loss_policy + loss_kl + loss_ent
        assert not np.isnan(total_loss.mean().item()) and not np.isinf(total_loss.mean().item()), \
            (loss_policy.mean().item(), loss_kl.mean().item())

        with torch.no_grad():
            if do_log:
                log_training_data(self._do_log, self.logger, self.frame_train, self._train_model, data)
                ratio = (data.logp - data.logp_policy).exp() - 1
                self.logger.add_scalar('ratio_mean' + tag, ratio.mean(), self.frame_train)
                self.logger.add_scalar('ratio_abs_mean' + tag, ratio.abs().mean(), self.frame_train)
                self.logger.add_scalar('ratio_abs_max' + tag, ratio.abs().max(), self.frame_train)
                self.logger.add_scalar('entropy' + tag, entropy.mean(), self.frame_train)
                self.logger.add_histogram('ratio_hist' + tag, ratio, self.frame_train)

        return total_loss

    def drop_collected_steps(self):
        self._prev_data = None

    def _log_set(self):
        self.logger.add_text(self.__class__.__name__, pprint.pformat(self._init_args))
        self.logger.add_text('Model', str(self._train_model))

    def __getstate__(self):
        d = dict(self.__dict__)
        d['_logger'] = None
        return d

    def __setstate__(self, d):
        self.__dict__ = d
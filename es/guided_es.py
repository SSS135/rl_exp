from copy import deepcopy

import numpy as np
import torch
import torch.autograd

from ppo_pytorch.algs.impala import IMPALA, AttrDict, copy_state_dict


class GuidedES(IMPALA):
    def __init__(self, *args,
                 grad_buffer_len=32,
                 steps_per_update=1,
                 es_lr=0.01,
                 es_std=0.01,
                 es_blend=0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_actors == 1
        assert steps_per_update == 1
        self.grad_buffer_len = grad_buffer_len
        self.steps_per_update = steps_per_update
        self.es_lr = es_lr
        self.es_std = es_std
        self.es_blend = es_blend

        # (k, n) grad history matrix
        self._grad_buffer = None
        self._grad_buffer_index = 0
        # up to 2 * P weight sets
        self._weights_to_test = []
        # up to 2 * P rewards
        self._es_rewards = []
        # (P, n) noise matrix
        self._noise = None
        self._orig_model_weights = deepcopy(list(self._train_model.state_dict().values()))

        # self.value_loss_scale = 0

    def _step(self, rewards, dones, states) -> torch.Tensor:
        reward = rewards[0] if rewards is not None else 0
        done = dones[0] if dones is not None else False

        actions = super()._step(rewards, dones, states)

        if len(self._es_rewards) != 0:
            # ent = self._eval_model.heads.logits.pd.entropy(torch.tensor(self._steps_processor.data.logits[-1])).item() if len(self._steps_processor.data.logits) != 0 else 0
            self._es_rewards[-1] += reward * self.reward_scale #+ self.entropy_reward_scale * ent

        if self._grad_buffer is not None and done:
            if len(self._weights_to_test) == 0:
                if len(self._es_rewards) != 0:
                    self._es_update()
                    self._es_rewards.clear()
                self._generate_weights_to_test()
            self._es_rewards.append(0)
            self._set_next_model_weights()

        return actions

    def _es_update(self):
        fitness = torch.tensor(self._es_rewards).view(self.steps_per_update, 2)
        fitness = fitness[:, 0] - fitness[:, 1]
        # (n) vector
        grad = self.es_lr / (2 * self.es_std * self.steps_per_update) * (fitness @ self._noise)
        # print('es grad', grad.pow(2).mean().sqrt(), fitness)
        weights = list(self._train_model.state_dict().values())
        w_lens = [w.numel() for w in weights]
        for curw, g, origw in zip(weights, grad.split(w_lens, 0), self._orig_model_weights):
            origw += g.view_as(origw)
            curw.copy_(origw)
        copy_state_dict(self._train_model, self._eval_model)

        if self._do_log:
            self.logger.add_scalar('es grad rms', grad.pow(2).mean().sqrt(), self.frame_train)
            self.logger.add_scalar('es buffer rms', self._grad_buffer.pow(2).mean().sqrt(), self.frame_train)
            self.logger.add_scalar('es fitness rms', fitness.pow(2).mean().sqrt(), self.frame_train)

    def _create_data(self):
        data = super()._create_data()
        data.state_values.fill_(0)
        return data

    def _impala_update(self, data: AttrDict):
        cur_weights = deepcopy(list(self._train_model.state_dict().values()))
        for src, dst in zip(self._orig_model_weights, self._train_model.state_dict().values()):
            dst.copy_(src)
        super()._impala_update(data)
        grads = []
        for prev, new in zip(self._orig_model_weights, self._train_model.state_dict().values()):
            grads.append(new - prev)
            new.copy_(prev)
        for src, dst in zip(cur_weights, self._train_model.state_dict().values()):
            dst.copy_(src)
        self._update_grad_buffer(grads)

    def _generate_weights_to_test(self):
        assert len(self._weights_to_test) == 0

        weights = list(self._orig_model_weights)
        w_sizes = [w.numel() for w in weights]

        self._noise = self._get_noise()

        for noises in zip(*self._noise.split(w_sizes, dim=1)):
            self._weights_to_test.append([w + e.view_as(w) for (e, w) in zip(noises, weights)])
            self._weights_to_test.append([w - e.view_as(w) for (e, w) in zip(noises, weights)])

    def _get_noise(self):
        """Generates (P, n) noise matrix"""
        buf_len, grad_len = self._grad_buffer.shape
        full_space_noise = self._grad_buffer.new_zeros(self.steps_per_update, grad_len).normal_()
        subspace_noise = self._grad_buffer.new_zeros(self.steps_per_update, buf_len).normal_() @ self._grad_buffer
        subspace_noise /= subspace_noise.pow(2).mean().sqrt()
        full_space_noise *= self.es_std * self.es_blend
        subspace_noise *= self.es_std * (1 - self.es_blend)
        # print('noises full', full_space_noise.pow(2).mean().sqrt(), 'ss', subspace_noise.pow(2).mean().sqrt())
        noise = full_space_noise + subspace_noise
        return noise

    def _set_next_model_weights(self):
        weights = self._weights_to_test.pop(0)
        for src, dst in zip(weights, self._train_model.state_dict().values()):
            dst.copy_(src)
        copy_state_dict(self._train_model, self._eval_model)

    def _update_grad_buffer(self, new_grads):
        new_grads = torch.cat([g.view(-1) for g in new_grads], dim=0)
        # print('algs grad', new_grads.pow(2).mean().sqrt())
        if self._grad_buffer is None:
            self._grad_buffer = new_grads.new_zeros((self.grad_buffer_len, new_grads.numel()))
        self._grad_buffer[self._grad_buffer_index] = new_grads
        self._grad_buffer_index += 1
        if self._grad_buffer_index >= self.grad_buffer_len:
            self._grad_buffer_index = 0
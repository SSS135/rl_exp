from functools import partial

import torch.optim as optim
from ppo_pytorch.common import DecayLR
from ppo_pytorch.actors import MLPActor, ActionValuesHead
from torch import nn as nn


def create_small_mlp_kwargs(dueling=True, activation=nn.Tanh, lr_decay_frames=1e5, **kwargs):
    defaults = dict(
        optim=partial(optim.Adam, lr=3e-4),
        eval_batch_size=1,
        replay_batch_size=32,
        trainable_batch_size=64,
        replay_size=50_000,
        target_net_update_freq=1000,
        eps=1,
        train_interval=4,
        reward_discount=0.995,
        model_factory=partial(MLPActor, activation=activation, head_factory=partial(ActionValuesHead, dueling=dueling)),
        cuda_eval=False,
        cuda_replay=False,
        double=True,
        learning_starts=2048,
        reward_scale=0.2,
        lr_scheduler=partial(DecayLR, start_value=1, end_value=0.05, end_epoch=lr_decay_frames),
    )
    defaults.update(kwargs)
    return defaults

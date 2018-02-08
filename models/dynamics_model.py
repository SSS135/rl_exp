from functools import partial
from itertools import chain

import torch
from rl.common.probability_distributions import make_pd
from torch import nn as nn
from torch.nn import functional as F


class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_space, hidden_size=128,
                 activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)):
        super().__init__()
        self.state_size = state_size
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.activation = activation
        self.pd = make_pd(action_space)

        self.gen_net = nn.Sequential(
            # prev state + action
            nn.Linear(state_size + self.pd.input_vector_len, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, state_size + 2)
            # next state + done + reward
        )
        self.disc_net_start = nn.Sequential(
            # prev state + action + next state + reward + done
            nn.Linear(state_size * 2 + self.pd.input_vector_len + 2, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.disc_net_end = nn.Sequential(
            activation(inplace=False),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        return self.generate(state, action)

    def generate(self, state, action):
        action_vec = self.pd.to_inputs(action)
        input = torch.cat([state, action_vec], 1)
        output = self.gen_net(input)
        done, reward, next_state = output[:, 0], output[:, 1], output[:, 2:]
        return next_state, reward.contiguous(), F.sigmoid(done)

    def discriminate(self, prev_state, next_state, action, reward, done):
        action_vec = self.pd.to_inputs(action)
        input = torch.cat([prev_state, next_state, action_vec, reward.view(-1, 1), done.view(-1, 1)], 1)
        hidden = self.disc_net_start(input)
        disc = self.disc_net_end(hidden)
        return disc.view(-1), hidden

    def params_disc(self):
        return chain(self.disc_net_start.parameters(), self.disc_net_end.parameters())

    def params_gen(self):
        return self.gen_net.parameters()
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import random


def create_mlp(input_size, output_size, layer_size, hidden_layers):
    net = []
    for i in range(hidden_layers):
        net.append(nn.Linear(input_size if i == 0 else layer_size, layer_size))
        net.append(nn.LeakyReLU)
    net.append(nn.Linear(layer_size if hidden_layers != 0 else input_size, output_size))
    return nn.Sequential(*net)


class MLP_G(nn.Module):
    def __init__(self, input_size, output_size, layer_size=128, hidden_layers=3):
        super(MLP_G, self).__init__()
        self.net = create_mlp(input_size, output_size, layer_size, hidden_layers)

    def forward(self, x):
        x = self.net(x)
        return x


class MLP_D(nn.Module):
    def __init__(self, input_size, layer_size=128, hidden_layers=3):
        super(MLP_D, self).__init__()
        self.net = create_mlp(input_size, 1, layer_size, hidden_layers)

    def forward(self, x):
        x = self.net(x)
        return x


# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0
#
#     def push(self, *data):
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = data
#         self.position = (self.position + 1) % self.capacity
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)


class VAE_LSGAN:
    def __init__(self, netG, netD, target_len, noise_len, condition_len=0):
        self.input_len = target_len
        self.condition_len = condition_len
        self.noise_len = noise_len
        self.step = 0
        self.netG = netG
        self.netD = netD

    def generate(self, condition=None):
        assert (condition is None) == (self.condition_len == 0)

    def process(self, input):
        pass



# class RBGAN:
#     def __init__(self, target_len, noise_len, condition_len=0, batch_size=256,
#                  train_interval=32, replay_buffer_len=50_000, training_starts=1024,
#                  optim_factory=partial(optim.Adam, lr=5e-5, betas=(0.5, 0.999)),
#                  G_factory=MLP_G, D_factory=MLP_D):
#         self.input_len = target_len
#         self.condition_len = condition_len
#         self.batch_size = batch_size
#         self.train_interval = train_interval
#         self.replay_buffer_len = replay_buffer_len
#         self.training_starts = training_starts
#         self.step = 0
#         self.netG = G_factory(num_input=noise_len+condition_len, num_output=target_len)
#         self.netD = D_factory(num_input=target_len+condition_len)
#         self.optimG = optim_factory(self.netG.parameters())
#         self.optimD = optim_factory(self.netD.parameters())
#         self.replay_buffer = ReplayBuffer(replay_buffer_len)
#
#     def append(self, target, condition=None):
#         assert (condition is None) == (self.condition_len == 0)
#         self.replay_buffer.push(target, condition)
#         if self.step >= self.training_starts and self.step % self.train_interval == 0:
#             self.train()
#         self.step += 1
#
#     def generate(self, condition=None):
#         assert (condition is None) == (self.condition_len == 0)
#
#     def train(self):
#         pass
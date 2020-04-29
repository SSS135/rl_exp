# http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import random
from collections import namedtuple

Transition = namedtuple('Transition', 'states, actions, next_states, rewards')


class ReplayBuffer:
    """
    Simple replay buffer for DQN.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, states, actions, next_states, rewards):
        """Saves a transition."""
        samples = [a for a in zip(states, actions, next_states, rewards)]
        for sample in samples:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = sample
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        return Transition(*zip(*samples))

    def __len__(self):
        return len(self.memory)
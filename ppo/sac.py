from .ppo import PPO
from ..models.heads import PolicyHead, StateValueHead


class SAC(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def head_factory(hidden_size, pd):
        return dict(probs=PolicyHead(hidden_size, pd), state_value=StateValueHead(hidden_size),
                    action_value=StateValueHead(hidden_size))



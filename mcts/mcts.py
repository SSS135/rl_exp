import numpy as np
# from mylib.optfn.kl_div import kl_div

from mylib.ppo_pytorch.common.rl_base import RLBase


class MCTS(RLBase):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

    def _step(self, prev_states, rewards, dones, cur_states) -> np.ndarray:
        pass

    @property
    def num_actors(self) -> int:
        raise NotImplementedError


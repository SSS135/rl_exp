import os
import random
from functools import partial
from multiprocessing.dummy import Pool

import gym
import gym.spaces
import numpy as np
from mlagents_envs.base_env import BatchedStepResult, AgentGroupSpec, ActionType
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel, EngineConfig
from ppo_pytorch.common.multiplayer_env import MultiplayerEnv
# from mlagents_envs import UnityWorkerInUseException
from ppo_pytorch.common.env_factory import NamedVecEnv, Monitor, ObservationNorm, ChannelTranspose
from ppo_pytorch.common.atari_wrappers import FrameStack
# from mlagents_envs.environment import UnityEnvironment, BrainInfo, BrainParameters, AllBrainInfo


DEFAULT_AGENT_GROUP = 'Player'


class UnityEnv(MultiplayerEnv):
    def __init__(self, env_name, visual_observations=False, train_mode=True, *args, **kwargs):
        self.visual_observations = visual_observations
        self.train_mode = train_mode
        engine_channel = EngineConfigurationChannel()
        engine_config = EngineConfig(80, 80, 1, 4.0, 30 * 4) if train_mode else EngineConfig(1280, 720, 1, 1.0, 60)
        engine_channel.set_configuration(engine_config)
        self._env = UnityEnvironment(env_name, *args, base_port=10000, side_channels=[engine_channel], **kwargs)
        self._env.reset()
        groups = self._env.get_agent_groups()
        self._agent_group = DEFAULT_AGENT_GROUP if DEFAULT_AGENT_GROUP in groups else groups[0]
        step_result = self._env.get_step_result(self._agent_group)
        agent_spec = self._env.get_agent_group_spec(self._agent_group)
        self.observation_space = self._get_observation_space(agent_spec)
        self.action_space = self._get_action_space(agent_spec)
        self.reward_range = (-1, 1)
        super().__init__(len(step_result.agent_id))

    def step(self, action) -> BatchedStepResult:
        self._env.set_actions(self._agent_group, action)
        self._env.step()
        res = self._env.get_step_result(self._agent_group)
        return self._process_brain_info(res)

    def reset(self) -> BatchedStepResult:
        self._env.reset()
        res = self._env.get_step_result(self._agent_group)
        return self._process_brain_info(res)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _process_brain_info(self, env_state: BatchedStepResult) -> BatchedStepResult:
        states = env_state.obs[0]
        if self.visual_observations:
            states = np.asarray(np.asarray(states) * 255, dtype=np.uint8)
        return env_state._replace(obs=states)

    def _get_action_space(self, spec: AgentGroupSpec):
        shape = spec.action_shape
        if spec.action_type == ActionType.CONTINUOUS:
            assert isinstance(spec.action_type, int)
            high = np.ones(shape)
            return gym.spaces.Box(-high, high)
        else:
            return gym.spaces.MultiDiscrete(shape)

    def _get_observation_space(self, spec: AgentGroupSpec):
        assert len(spec.observation_shapes) == 1
        if self.visual_observations:
            zeros = np.zeros(spec.observation_shapes[0])
            return gym.spaces.Box(zeros, zeros + 255, dtype=np.uint8)
        else:
            ones = np.ones(spec.observation_shapes[0])
            return gym.spaces.Box(-ones, ones, dtype=np.float32)

    def close(self):
        self._env.close()


class UnityVecEnv(NamedVecEnv):
    def __init__(self, env_path, parallel='thread', visual_observations=False, observation_norm=False,
                 stacked_frames=1, train_mode=True):
        self.env_path = env_path
        self.visual_observations = visual_observations
        self.train_mode = train_mode
        env_name = os.path.basename(os.path.split(os.path.normpath(env_path))[0])
        self.observation_norm = observation_norm
        self.stacked_frames = stacked_frames
        self.env_name = env_name
        self.parallel = parallel
        self.envs = None
        self.num_envs = None
        self._pool = None

        env: UnityEnv = self.get_env_factory()()
        data = env.reset()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._initial_num_players = len(data.agent_id)
        env.close()

    def get_env_factory(self):
        def make(env_path, observation_norm, visual_observations, stacked_frames, train_mode):
            while True:
                worker_id = random.randrange(50000)
                try:
                    env = UnityEnv(env_path, worker_id=worker_id, visual_observations=visual_observations,
                                   train_mode=train_mode)
                    break
                except UnityWorkerInUseException:
                    print(f'Worker {worker_id} already in use')
            env = Monitor(env)
            if visual_observations:
                env = ChannelTranspose(env)
            if stacked_frames > 1:
                assert visual_observations
                env = FrameStack(env, 4)
            if observation_norm:
                env = ObservationNorm(env)
            return env

        return partial(make, self.env_path, self.observation_norm, self.visual_observations, self.stacked_frames, self.train_mode)

    def set_num_envs(self, num_envs):
        if self.envs is not None:
            for env in self.envs:
                env.close()
        self.num_envs = num_envs
        env_factory = self.get_env_factory()
        self._pool = Pool(self.num_envs)
        self.envs = self._pool.map(lambda _: env_factory(), range(num_envs))

    def step(self, actions) -> BatchedStepResult:
        actions = np.split(np.asarray(actions), len(self.envs))
        data = self._pool.starmap(lambda env, a: env.step(a), zip(self.envs, actions))
        return BatchedStepResult._make([np.concatenate(x, 0) for x in zip(*[tuple(t) for t in data])])

    def reset(self) -> BatchedStepResult:
        data = self._pool.starmap(lambda env: env.reset(), zip(self.envs))
        return BatchedStepResult._make([np.concatenate(x, 0) for x in zip(*[tuple(t) for t in data])])

    def __enter__(self):
        self._pool.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.map(lambda env: env.close(), self.envs)
        self._pool.__exit__()


# class SimplifiedUnityVecEnv(UnityVecEnv):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.num_players = self._initial_num_players
#         self._index_to_id = {}
#         self._prev_ids = []
#
#     def set_num_envs(self, num_envs):
#         super().set_num_envs(num_envs // self.num_players)
#
#     def step(self, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
#         res = super().step(actions)
#         if len(self._prev_ids) != len(res.agent_id):
#
#         return res.obs, res.reward, res.done, [dict() for _ in range(self.num_players)]
#
#     def reset(self) -> np.ndarray:
#         res = super().reset()
#         self._index_to_id = {index: id for id, index in res.agent_id_to_index.items()}
#         self._prev_ids = res.agent_id
#         return res.obs
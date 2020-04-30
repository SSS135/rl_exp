import os
import random
from enum import Enum
from functools import partial
from typing import NamedTuple, Any, List, Optional

import numpy as np
from gym_unity.envs import UnityEnv
from mlagents_envs.exception import UnityWorkerInUseException
from ppo_pytorch.common.atari_wrappers import FrameStack
from ppo_pytorch.common.env_factory import Monitor, ObservationNorm, ChannelTranspose
import multiprocessing as mp


class Command(Enum):
    stats = 1
    reset = 3
    step = 2
    shutdown = 4


class Message(NamedTuple):
    command: Command
    payload: Any


def process_entry(pipe: mp.connection.Connection, env_fn):
    env: UnityEnv = env_fn()
    env.reset()
    while True:
        msg: Message = pipe.recv()
        if msg.command == Command.stats:
            pipe.send((env.observation_space, env.action_space, env.number_agents))
        elif msg.command == Command.reset:
            pipe.send(env.reset())
        elif msg.command == Command.step:
            pipe.send(env.step(msg.payload))
        elif msg.command == Command.shutdown:
            env.close()
            pipe.send(True)
            return


def make_env(env_path, observation_norm, visual_observations, stacked_frames, no_graphics):
    while True:
        worker_id = random.randrange(50000)
        try:
            env = UnityEnv(env_path, worker_id=worker_id, no_graphics=no_graphics,
                           uint8_visual=visual_observations, use_visual=visual_observations)
            break
        except UnityWorkerInUseException:
            print(f'Worker {worker_id} already in use')
    env = Monitor(env)
    if visual_observations:
        env = ChannelTranspose(env)
    if stacked_frames > 1:
        assert visual_observations
        env = FrameStack(env, stacked_frames)
    if observation_norm:
        env = ObservationNorm(env)
    return env


class UnityVecEnv:
    def __init__(self, env_path, visual_observations=False, observation_norm=False, stacked_frames=1):
        self.env_path = env_path
        self.visual_observations = visual_observations
        self.observation_norm = observation_norm
        self.stacked_frames = stacked_frames

        self.env_name = os.path.basename(os.path.split(os.path.normpath(env_path))[0])
        self.num_actors = None
        self._processes: List[mp.Process] = []
        self._pipes: List[mp.connection.Connection] = []

        self._create_process()
        self.observation_space, self.action_space, self.actors_per_env = self._send_message(Command.stats)[0]

    def _create_process(self):
        agent_id = len(self._processes)
        parent_conn, child_conn = mp.Pipe()
        env_factory = partial(self.get_env_factory(), no_graphics=False)
        proc = mp.Process(name=f'agent {agent_id}', target=process_entry, args=(child_conn, env_factory))
        proc.start()
        self._processes.append(proc)
        self._pipes.append(parent_conn)

    def get_env_factory(self):
        return partial(make_env, self.env_path, self.observation_norm, self.visual_observations, self.stacked_frames)

    def set_num_actors(self, num_actors):
        assert num_actors >= len(self._processes) * self.actors_per_env
        assert num_actors % self.actors_per_env == 0, (num_actors, self.actors_per_env)

        self.num_actors = num_actors
        num_envs = self.num_actors // self.actors_per_env
        while len(self._processes) < num_envs:
            self._create_process()

    def step(self, actions):
        actions = np.split(np.asarray(actions), len(self._processes))
        data = self._send_message(Command.step, actions)
        return [np.stack(x, 0) for x in zip(*data)]

    def reset(self):
        data = self._send_message(Command.reset)
        return np.stack(data, 0)

    def _send_message(self, command: Command, payload: Optional[List] = None) -> List:
        if payload is None:
            payload = len(self._processes) * [None]
        for pipe, payload in zip(self._pipes, payload):
            pipe.send(Message(command, payload))
        return [pipe.recv() for pipe in self._pipes]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._send_message(Command.shutdown)
        for proc in self._processes:
            proc.join(30)
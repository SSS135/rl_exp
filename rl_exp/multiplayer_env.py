import gym


class MultiplayerEnv(gym.Env):
    def __init__(self, num_players):
        self.num_players = num_players
import numpy as np
import gym


class BitsGame(gym.Env):

    def __init__(self, size=8, shaped_reward=False):
        self.size = size
        self.shaped_reward = shaped_reward
        self.action_space = gym.spaces.Discrete(size)
        self.observation_space = gym.spaces.Box(low=np.array([0] * size), high=np.array([1] * size))
        # self.state = np.random.randint(2, size=size)
        # self.target = np.random.randint(2, size=size)
        # while np.sum(self.state == self.target) == size:
        #     self.target = np.random.randint(2, size=size)

    def step(self, action):
        self.state[action] = 1 - self.state[action]
        if self.shaped_reward:
            return np.copy(self.state), -np.sum(np.square(self.state - self.target))
        else:
            if not np.sum(self.state == self.target) == self.size:
                return np.copy(self.state), -1, False, {}
            else:
                return np.copy(self.state), 0, True, {}

    def reset(self, size=None):
        if size is None:
            size = self.size
        self.state = np.random.randint(2, size=size)
        self.target = np.random.randint(2, size=size)
        return self.state, self.target

    def render(self, mode='human'):
        raise NotImplementedError

import gym
import numpy as np
import random

class StochasticMDPEnv(gym.Env):
    """
    the state is represented as 0, 1, 2, 3, 4, 5
    the action is represented as 0(left), 1(right)
    if choose right, there is a probability of RIGHT_PROB to go right,
    if choose left, there is a probability of 100 to go left.
    """
    RIGHT_PROB = 0.5

    def __init__(self):
        self.visited_six = False
        self.current_state = 1

        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([5]))
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        self.visited_six = False
        self.current_state = 1
        return np.array([self.current_state])

    def step(self, action):
        if self.current_state != 0:
            # If "right" selected
            if action == 1:
                if random.random() < self.RIGHT_PROB and self.current_state < 5:
                    self.current_state += 1
                else:
                    self.current_state -= 1
            # If "left" selected
            if action == 0:
                self.current_state -= 1
            # If state 6 reached
            if self.current_state == 5:
                self.visited_six = True
        if self.current_state == 0:
            if self.visited_six:
                return np.array([self.current_state]), 1.00, True, {}
            else:
                return np.array([self.current_state]), 1.00/100.00, True, {}
        else:
            return np.array([self.current_state]), 0.0, False, {}

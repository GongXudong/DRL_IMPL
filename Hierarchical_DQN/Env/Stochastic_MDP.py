import gym
import numpy as np
import random

class StochasticMDPEnv(gym.Env):

    def __init__(self):
        self.visited_six = False
        self.current_state = 2

        self.observation_space = gym.spaces.Box(low=np.array([1]), high=np.array([6]))
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        self.visited_six = False
        self.current_state = 2
        return np.array([self.current_state])

    def step(self, action):
        if self.current_state != 1:
            # If "right" selected
            if action == 1:
                if random.random() < 0.5 and self.current_state < 6:
                    self.current_state += 1
                else:
                    self.current_state -= 1
            # If "left" selected
            if action == 0:
                self.current_state -= 1
            # If state 6 reached
            if self.current_state == 6:
                self.visited_six = True
        if self.current_state == 1:
            if self.visited_six:
                return np.array([self.current_state]), 1.00, True, {}
            else:
                return np.array([self.current_state]), 1.00/100.00, True, {}
        else:
            return np.array([self.current_state]), 0.0, False, {}

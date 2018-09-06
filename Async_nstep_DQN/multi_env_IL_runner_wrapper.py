
import gym
import numpy as np

from Env.cmotp_IL import CMOTP

class MultiEnvRunnerWrapper(gym.Wrapper):

    def __init__(self, num_env, env_class):
        self.num_env = num_env
        self.envs = [env_class() for _ in range(num_env)]

    def reset(self, env_list=None):
        if env_list is None:
            env_list = list(range(0, self.num_env))

        states_1 = []
        states_2 = []
        for i in env_list:
            assert i >=0 and i < self.num_env, 'env_list has number that does not in range(0, num_env)!'
            state_1, state_2 = self.envs[i].reset()
            states_1.append(state_1)
            states_2.append(state_2)

        return states_1, states_2


if __name__ == '__main__':

    multienv = MultiEnvRunnerWrapper(4, CMOTP)
    print(multienv.envs)
    multienv.envs[0].reset()
    print(multienv.envs[0].step([2, 4]))

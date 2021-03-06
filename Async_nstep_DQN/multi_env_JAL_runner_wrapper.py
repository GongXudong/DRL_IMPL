
import gym
import numpy as np

from Env.cmotp_JAL import CMOTP

class MultiEnvRunnerWrapper(gym.Wrapper):

    def __init__(self, num_env, env_class):
        self.num_env = num_env
        self.envs = [env_class() for _ in range(num_env)]

    def reset(self, env_list=None):
        if env_list is None:
            env_list = list(range(0, self.num_env))

        states = []
        for i in env_list:
            assert i >=0 and i < self.num_env, 'env_list has number that does not in range(0, num_env)!'
            states.append(self.envs[i].reset())

        return states

    def step(self, actions):
        assert len(actions) == self.num_env, "action number should equal num_env!"
        res = [self.envs[i].step(action) for i, action in enumerate(actions)]
        obs, rwds, dns, infs = zip(*res)
        return np.stack(obs), np.stack(rwds), np.stack(dns), infs


if __name__ == '__main__':

    multienv = MultiEnvRunnerWrapper(4, CMOTP)
    print(multienv.envs)
    multienv.envs[0].reset()
    print(multienv.envs[0].step([2, 4]))

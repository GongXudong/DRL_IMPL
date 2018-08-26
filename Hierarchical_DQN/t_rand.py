import time
import numpy as np
from Env.Stochastic_MDP import StochasticMDPEnv

if __name__ == '__main__':

    env = StochasticMDPEnv()

    episode_lens = []
    episode_rewards = []

    for i in range(10000):
        state = env.reset()
        episode_len = 0
        episode_reward = 0
        while True:
            action = np.random.choice([0, 1])
            # action = 1
            next_state, reward, done, _ = env.step(action)
            episode_len += 1
            episode_reward += reward
            if done:
                break
        episode_lens.append(episode_len)
        episode_rewards.append(episode_reward)
        if i % 1000 == 999:
            print('averaged lens: {}    averaged rewards: {}'.format(np.mean(episode_lens[-1000:]),
                                                                     np.mean(episode_rewards[-1000:])))
            time.sleep(2)


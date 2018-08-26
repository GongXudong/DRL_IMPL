import time
import numpy as np
from DQN.DQNAgent import DQNAgent
from Env.Stochastic_MDP import StochasticMDPEnv

if __name__ == '__main__':

    env = StochasticMDPEnv()
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, [32, 32, 32], 'smdp',
                     epsilon_decay_step=10000, epsilon_end=0.02, replay_memory_size=50000,
                     learning_rate=5e-4)

    episode_lens = []
    episode_rewards = []

    for i in range(100000):
        state = env.reset()
        episode_len = 0
        episode_reward = 0
        while True:
            action = agent.choose_action(state=state)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, float(done))
            agent.train()

            episode_len += 1
            episode_reward += reward

            state = next_state

            if done:
                # print('episode_{}: len {}  reward {}'.format(i, episode_len, episode_reward))
                break

        episode_lens.append(episode_len)
        episode_rewards.append(episode_reward)
        if i % 1000 == 999:
            print('averaged lens: {}    averaged rewards: {}'.format(np.mean(episode_lens[-1000:]),
                                                                     np.mean(episode_rewards[-1000:])))
            time.sleep(2)
    env.close()
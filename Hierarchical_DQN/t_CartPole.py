import gym
import time
import numpy as np
import tensorflow as tf
from Hierarchical_DQN.DQNAgent import DQNAgent

DEBUG = False

if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    print((None,) + env.observation_space.shape, env.action_space.n)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, [64], 'cartpole',
                     epsilon_decay_step=10000, epsilon_end=0.02, replay_memory_size=50000,
                     learning_rate=5e-4)

    for i in range(10000):
        state = env.reset()
        episode_len = 0
        episode_reward = 0
        while True:
            if DEBUG:
                env.render()
            action = agent.choose_action(state=state)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, float(done))
            agent.train()

            episode_len += 1
            episode_reward += reward

            state = next_state

            if done:
                print('episode_{}: len {}  reward {}'.format(i, episode_len, episode_reward))
                break

    env.close()
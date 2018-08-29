
from Env.cmotp import CMOTP
from DQN.DQNAgent import DQNAgent
from Lenient.temperature_record import Temp_record
import time

DEBUG = False

TS_GREEDY_COEFF = 1.0

TRAIN = False
TEST_NUM = 10

if __name__ == '__main__':

    env = CMOTP()
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, [512, 512], 'cmotp',
                     epsilon_decay_step=10000, epsilon_end=0.05, replay_memory_size=100000,
                     learning_rate=1e-4, targetnet_update_freq=5000)
    if TRAIN:
        temp_record = Temp_record(shape=tuple(env.observation_space.high + 1) + (env.action_space.n, ), beta_len=1500)

        for i in range(5000):
            state = env.reset()
            episode_len = 0
            episode_reward = 0
            episode = []
            while True:
                if DEBUG:
                    env.render()
                action = agent.choose_action(state=state, epsilon=pow(temp_record.get_state_temp(state), TS_GREEDY_COEFF))
                action_n = [int(action % 5), int(action / 5)]
                next_state, reward, done, _ = env.step(action_n)
                agent.store(state, action, reward, next_state, float(done))

                episode.append((state, action))

                agent.train()

                episode_len += 1
                episode_reward += reward

                state = next_state

                if done:
                    print('episode_{}: len {}  reward {}'.format(i, episode_len, episode_reward))
                    break

            #temp decay
            temp_record.decay_temp(episode)

    else:
        # test
        agent.load_model()
        for i in range(TEST_NUM):
            state = env.reset()
            while True:
                env.render()
                time.sleep(2)
                action = agent.choose_action(state, epsilon=0.05)
                action_n = [int(action % 5), int(action / 5)]
                next_state, reward, done, _ = env.step(action_n)
                state = next_state
                if done:
                    break


    env.close()





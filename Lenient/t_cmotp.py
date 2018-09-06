
from Env.cmotp_IL import CMOTP
from Lenient.lenient_DQNAgent import LenientDQNAgent
import numpy as np
import time

TRAIN_EPISODES = 5000

TRAIN = True
TRAIN = False

TEST_NUM = 10


if __name__ == '__main__':
    env = CMOTP()
    agent1 = LenientDQNAgent(env, [256, 256], 'LenientAgent1',
                             learning_rate=1e-4, replay_memory_size=100000,
                             use_tau=True, tau=1e-3,
                             logdir='logs1', savedir='save1')

    agent2 = LenientDQNAgent(env, [256, 256], 'LenientAgent2',
                             learning_rate=1e-4, replay_memory_size=100000,
                             use_tau=True, tau=1e-3,
                             logdir='logs2', savedir='save2')
    print('after init')

    if TRAIN:

        for i in range(TRAIN_EPISODES):
            state1, state2 = env.reset()
            episode_len = 0
            episode1, episode2 = [], []

            while True:
                action1 = agent1.choose_action(state1)
                action2 = agent2.choose_action(state2)
                next_state, reward, done, _ = env.step([action1, action2])
                next_state1, next_state2 = next_state
                reward1, reward2 = reward
                done1, done2 = done
                leniency1 = agent1.leniency_calculator.calc_leniency(agent1.temp_recorder.get_state_action_temp(state1, action1))
                leniency2 = agent2.leniency_calculator.calc_leniency(agent2.temp_recorder.get_state_action_temp(state2, action2))

                # print('state: ', state1, state2)
                # print('action: ', action1, action2)
                # print('reward: ', reward1, reward2)
                # print('done: ', done1, done2)
                # print('next_state: ', next_state1, next_state2)
                # print('leniencies: ', leniency1, leniency2)

                agent1.store(state1, action1, reward1, next_state1, done1, leniency1)
                agent2.store(state2, action2, reward2, next_state2, done2, leniency2)

                episode1.append((state1, action1))
                episode2.append((state2, action2))

                agent1.train()
                agent2.train()

                episode_len += 1

                state1 = next_state1
                state2 = next_state2

                if done1:
                    print('episode_{}, len: {}'.format(i, episode_len))
                    break

            agent1.temp_recorder.decay_temp(episode1)
            agent2.temp_recorder.decay_temp(episode2)
    else:
        agent1.load_model()
        agent2.load_model()
        for i in range(TEST_NUM):
            state1, state2 = env.reset()
            while True:
                env.render()
                time.sleep(1)
                action1 = agent1.choose_action(state1, 0.0)
                action2 = agent2.choose_action(state2, 0.0)
                next_state, reward, done, _ = env.step([action1, action2])
                next_state1, next_state2 = next_state
                state1 = next_state1
                state2 = next_state2
                if done[0]:
                    break

    env.close()

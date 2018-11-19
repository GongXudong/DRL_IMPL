
from Env.cmotp_IL import CMOTP
from Lenient.lenient_DQNAgent import LenientDQNAgent
import numpy as np
import tensorflow as tf
import random
import time

TRAIN_EPISODES = 5000

TRAIN = True
# TRAIN = False

TEST_NUM = 10


def set_seed(lucky_number):
    tf.set_random_seed(lucky_number)
    np.random.seed(lucky_number)
    random.seed(lucky_number)


def test(ag1, ag2, render=False, load_model=False):
    test_env = CMOTP()
    if load_model:
        ag1.load_model()
        ag2.load_model()
    ep_log = []
    for iii in range(TEST_NUM):
        state1, state2 = test_env.reset()
        ep_len = 0
        while True:
            if render:
                test_env.render()
                time.sleep(1)
            action1 = ag1.choose_action(state1, 0.0)
            action2 = ag2.choose_action(state2, 0.0)
            next_state_, reward_, done_, _ = test_env.step([action1, action2])
            next_state1, next_state2 = next_state_
            state1 = next_state1
            state2 = next_state2
            ep_len += 1
            if done_[0] or ep_len >= 1000:
                ep_log.append(ep_len)
                break
    return np.mean(ep_log)


def main():
    env = CMOTP()

    lucky_no = 5
    set_seed(lucky_no)

    erm_size = 10000

    agent1 = LenientDQNAgent(env, [256, 256], 'LenientAgent1',
                             learning_rate=1e-4, replay_memory_size=erm_size,
                             use_tau=True, tau=1e-3,
                             logdir='logs1', savedir='save1',
                             batch_size=50)

    agent2 = LenientDQNAgent(env, [256, 256], 'LenientAgent2',
                             learning_rate=1e-4, replay_memory_size=erm_size,
                             use_tau=True, tau=1e-3,
                             logdir='logs2', savedir='save2',
                             batch_size=50)
    print('after init')
    begintime = time.time()

    if TRAIN:

        episodes_recorder = []
        train_log = []
        train_num = 0

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
                leniency1 = agent1.leniency_calculator.calc_leniency(
                    agent1.temp_recorder.get_state_action_temp(state1, action1))
                leniency2 = agent2.leniency_calculator.calc_leniency(
                    agent2.temp_recorder.get_state_action_temp(state2, action2))

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
                train_num += 1

                episode_len += 1

                state1 = next_state1
                state2 = next_state2

                if done1:
                    this_train_log = (train_num, i, agent1.temp_recorder.get_ave_temp(), episode_len)
                    train_log.append(this_train_log)
                    print('train_cnt: {}, episode: {}, ave_temp: {}, len: {}'.format(*this_train_log))
                    episodes_recorder.append(episode_len)

                    agent1.temp_recorder.show_temp(big=True, narrow=True)

                    if i > 0 and i % 100 == 0:
                        print('testing...')
                        print('average episode length: ', test(agent1, agent2, render=False, load_model=False))

                    break

            agent1.temp_recorder.decay_temp(episode1)
            agent2.temp_recorder.decay_temp(episode2)

        endtime = time.time()
        print('training time: {}'.format(endtime - begintime))

        # np.save('{}-{}.npy'.format(erm_size, lucky_no), episodes_recorder)
        np.save('train_log_{}_{}.npy'.format(erm_size, lucky_no), train_log)
    else:
        test(agent1, agent2, render=True, load_model=True)

    env.close()


if __name__ == '__main__':
    main()

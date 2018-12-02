import sys

sys.path.append('..')

from Env.cmotp_IL import CMOTP
from Async_nstep_DQN.lenient_DQNAgent_for_multi_envs import LenientDQNAgent
from Async_nstep_DQN.multi_env_IL_runner_wrapper import MultiEnvRunnerWrapper
from common.replay_buffer import ReplayBuffer
import numpy as np
import time
import tensorflow as tf
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_num', help='environment number', type=int, default=10)
parser.add_argument('--step_len', help='n_step length', type=int, default=5)
parser.add_argument('--random_seed', help='random seed', type=int, default=9999)
parser.add_argument('--max_u', help='max u', type=float, default=0.995)
parser.add_argument('--train_times', help='train times', type=int, default=1000)
parser.add_argument('--train', help='train(1) or test(0)', type=int, default=1, choices=[0, 1])
args = parser.parse_args()

DEBUG = False

TS_GREEDY_COEFF = 1.0

TRAIN = bool(args.train)
# TRAIN = False

GAMMA = 0.95

STEP_N = args.step_len
ENV_NUM = args.env_num
RANDOM_SEED = args.random_seed

ERM_FACTOR = 10
ERM_TRAIN_NUM = 5

TEST_NUM = 10
TRAIN_TIMES = args.train_times
MAX_U = args.max_u
TRAIN_EPISODES = TRAIN_TIMES * ENV_NUM

print('step_n: {}, env_num: {}'.format(STEP_N, ENV_NUM))


def set_seed(lucky_number):
    tf.set_random_seed(lucky_number)
    np.random.seed(lucky_number)
    random.seed(lucky_number)


def discount_with_dones(rewards, dones, gamma):
    """
    return n_step TD_values
    :param rewards:
    :param dones:
    :param gamma:
    :return:
    """
    discounted = []
    r = 0
    for rw, dn in zip(rewards[::-1], dones[::-1]):
        r = rw + gamma * r * (1. - dn)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


def test(ag1, ag2, render=False, load_model=False):
    test_env = CMOTP()
    if load_model:
        ag1.load_model()
        ag2.load_model()
    ep_log = []
    for iii in range(TEST_NUM):
        print('test {}'.format(iii))
        state1, state2 = test_env.reset()
        ep_len = 0
        while True:
            if render:
                test_env.render()
                time.sleep(1)
            action1 = ag1.choose_action(state1, 0, 0.0)
            action2 = ag2.choose_action(state2, 0, 0.0)
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
    env = MultiEnvRunnerWrapper(ENV_NUM, CMOTP)

    lucky_no = RANDOM_SEED
    set_seed(lucky_no)

    agent1 = LenientDQNAgent(env.envs[0], ENV_NUM, [256, 256], 'LenientAgent1',
                             learning_rate=1e-4,
                             use_tau=True, tau=1e-3,
                             mu=MAX_U,
                             logdir='logs/logs1_ERM_{}_{}_{}_{}'.format(ENV_NUM, STEP_N, RANDOM_SEED, TRAIN_TIMES),
                             savedir='save/save1_ERM_{}_{}_{}_{}'.format(ENV_NUM, STEP_N, RANDOM_SEED, TRAIN_TIMES),
                             auto_save=False, discount=GAMMA)

    agent2 = LenientDQNAgent(env.envs[0], ENV_NUM, [256, 256], 'LenientAgent2',
                             learning_rate=1e-4,
                             use_tau=True, tau=1e-3,
                             mu=MAX_U,
                             logdir='logs/logs2_ERM_{}_{}_{}_{}'.format(ENV_NUM, STEP_N, RANDOM_SEED, TRAIN_TIMES),
                             savedir='save/save2_ERM_{}_{}_{}_{}'.format(ENV_NUM, STEP_N, RANDOM_SEED, TRAIN_TIMES),
                             auto_save=False, discount=GAMMA)
    erm1 = ReplayBuffer(ERM_FACTOR * ENV_NUM * STEP_N)
    erm2 = ReplayBuffer(ERM_FACTOR * ENV_NUM * STEP_N)

    print('after init')
    begintime = time.time()

    if TRAIN:

        train_input_shape = (ENV_NUM * STEP_N,) + env.envs[0].observation_space.shape

        episodes_1 = [[] for _ in range(ENV_NUM)]
        episodes_2 = [[] for _ in range(ENV_NUM)]
        states_1, states_2 = env.reset()

        ep_cnt = 0

        ep_len_log = []
        min_len = 10000.

        train_num = 0
        train_log = []

        # 针对某一状态，记录所有环境中，该状态的温度值
        temp_log = [[] for _ in range(ENV_NUM)]

        while len(ep_len_log) < TRAIN_EPISODES:

            sts_1 = [[] for _ in range(ENV_NUM)]
            acts_1 = [[] for _ in range(ENV_NUM)]
            rwds_1 = [[] for _ in range(ENV_NUM)]
            n_sts_1 = [[] for _ in range(ENV_NUM)]
            dns_1 = [[] for _ in range(ENV_NUM)]
            ln_1 = [[] for _ in range(ENV_NUM)]

            sts_2 = [[] for _ in range(ENV_NUM)]
            acts_2 = [[] for _ in range(ENV_NUM)]
            rwds_2 = [[] for _ in range(ENV_NUM)]
            n_sts_2 = [[] for _ in range(ENV_NUM)]
            dns_2 = [[] for _ in range(ENV_NUM)]
            ln_2 = [[] for _ in range(ENV_NUM)]

            # get a batch of train data
            for j in range(ENV_NUM):
                for k in range(STEP_N):
                    action_1 = agent1.choose_action(states_1[j], j)
                    action_2 = agent2.choose_action(states_2[j], j)
                    action_n = [action_1, action_2]
                    next_state, reward, done, _ = env.envs[j].step([action_1, action_2])
                    next_state_1, next_state_2 = next_state
                    reward_1, reward_2 = reward
                    done_1, done_2 = done

                    episodes_1[j].append((states_1[j], action_1))
                    episodes_2[j].append((states_2[j], action_2))

                    sts_1[j].append(states_1[j])
                    acts_1[j].append(action_1)
                    rwds_1[j].append(reward_1)
                    n_sts_1[j].append(next_state_1)
                    dns_1[j].append(done_1)
                    ln_1[j].append(
                        agent1.leniency_calculator.calc_leniency(agent1.temp_recorders[j].get_state_temp(states_1[j])))

                    sts_2[j].append(states_2[j])
                    acts_2[j].append(action_2)
                    rwds_2[j].append(reward_2)
                    n_sts_2[j].append(next_state_2)
                    dns_2[j].append(done_2)
                    ln_2[j].append(
                        agent2.leniency_calculator.calc_leniency(agent2.temp_recorders[j].get_state_temp(states_2[j])))

                    states_1[j] = next_state_1
                    states_2[j] = next_state_2

                    if done_1:
                        states_1[j], states_2[j] = env.envs[j].reset()
                        agent1.temp_recorders[j].decay_temp(episodes_1[j])
                        agent2.temp_recorders[j].decay_temp(episodes_2[j])

                        ep_cnt += 1

                        this_train_log = (train_num, ep_cnt, j,
                                          agent1.temp_recorders[j].get_ave_temp(),
                                          agent1.temp_recorders[j].get_temp_len(),
                                          len(episodes_1[j]))
                        train_log.append(this_train_log)

                        print('train_num: {}, episode_cnt: {}, env: {} , mean_temp: {}, temp_len: {}, len: {} '.format(
                            *this_train_log))
                        checked_temp = agent1.temp_recorders[j].show_temp(big=True, narrow=False)
                        temp_log[j].append(checked_temp)

                        if ep_cnt % 100 == 0:
                            print('testing...')
                            print('average episode length: ', test(agent1, agent2, render=False, load_model=False))

                        ep_len_log.append(len(episodes_1[j]))
                        tmp = np.mean(ep_len_log[-10:])
                        if tmp < min_len:
                            print('update min_len with ', tmp)
                            min_len = tmp
                            agent1.save_model()
                            agent2.save_model()

                        episodes_1[j] = []
                        episodes_2[j] = []

            # discount reward
            last_values_1 = agent1.get_max_target_Q_s_a(states_1)
            for j, (rwd_j, dn_j, l_v_j) in enumerate(zip(rwds_1, dns_1, last_values_1)):
                if type(rwd_j) is np.ndarray:
                    rwd_j = rwd_j.tolist()
                if type(dn_j) is np.ndarray:
                    dn_j = dn_j.tolist()

                if dn_j[-1] == 0:
                    rwd_j = discount_with_dones(rwd_j + [l_v_j], dn_j + [0], GAMMA)[:-1]
                else:
                    rwd_j = discount_with_dones(rwd_j, dn_j, GAMMA)

                rwds_1[j] = rwd_j

            last_values_2 = agent2.get_max_target_Q_s_a(states_2)
            for j, (rwd_j, dn_j, l_v_j) in enumerate(zip(rwds_2, dns_2, last_values_2)):
                if type(rwd_j) is np.ndarray:
                    rwd_j = rwd_j.tolist()
                if type(dn_j) is np.ndarray:
                    dn_j = dn_j.tolist()

                if dn_j[-1] == 0:
                    rwd_j = discount_with_dones(rwd_j + [l_v_j], dn_j + [0], GAMMA)[:-1]
                else:
                    rwd_j = discount_with_dones(rwd_j, dn_j, GAMMA)

                rwds_2[j] = rwd_j

            # flatten
            sts_1 = np.asarray(sts_1, dtype=np.float32).reshape(train_input_shape)
            acts_1 = np.asarray(acts_1, dtype=np.int32).flatten()
            rwds_1 = np.asarray(rwds_1, dtype=np.float32).flatten()
            n_sts_1 = np.asarray(n_sts_1, dtype=np.float32).reshape(train_input_shape)
            dns_1 = np.asarray(dns_1, dtype=np.bool).flatten()
            ln_1 = np.asarray(ln_1, dtype=np.float32).flatten()

            sts_2 = np.asarray(sts_2, dtype=np.float32).reshape(train_input_shape)
            acts_2 = np.asarray(acts_2, dtype=np.int32).flatten()
            rwds_2 = np.asarray(rwds_2, dtype=np.float32).flatten()
            n_sts_2 = np.asarray(n_sts_2, dtype=np.float32).reshape(train_input_shape)
            dns_2 = np.asarray(dns_2, dtype=np.bool).flatten()
            ln_2 = np.asarray(ln_2, dtype=np.float32).flatten()

            # train
            agent1.train_without_replaybuffer(sts_1, acts_1, rwds_1, ln_1)
            agent2.train_without_replaybuffer(sts_2, acts_2, rwds_2, ln_2)
            train_num += 1


            # store these transitions to ERM
            for ii, (s1, a1, td1, l1) in enumerate(zip(sts_1, acts_1, rwds_1, ln_1)):
                erm1.add(s1, a1, td1, [], l1)
            for ii, (s2, a2, td2, l2) in enumerate(zip(sts_2, acts_2, rwds_2, ln_2)):
                erm2.add(s2, a2, td2, [], l2)

            # print(sts_1)
            # print(acts_1)
            # print(rwds_1)
            # print(ln_1)
            # print('----------------------')
            # erm1.show()
            # exit()

            # train with transitions from ERM
            for ii in range(ERM_TRAIN_NUM):
                erm_s1, erm_a1, erm_td1, _, erm_l1 = erm1.sample(ENV_NUM * STEP_N)
                erm_s2, erm_a2, erm_td2, _, erm_l2 = erm2.sample(ENV_NUM * STEP_N)
                # print('*************************')
                # print(erm_s1)
                # print(erm_a1)
                # print(erm_td1)
                # print(erm_l1)
                # exit()
                agent1.train_without_replaybuffer(erm_s1, erm_a1, erm_td1, erm_l1)
                agent2.train_without_replaybuffer(erm_s2, erm_a2, erm_td2, erm_l2)
                train_num += 1


        endtime = time.time()
        print('training time: {}'.format(endtime - begintime))

        with open('./train_log.txt', 'a') as f:
            f.write('ERM num_env: {}, n_step: {}, rand_seed: {}, episodes: {}, training time: {}'.format(
                ENV_NUM, STEP_N, RANDOM_SEED, TRAIN_TIMES, endtime - begintime) + '\n')

        # np.save('ep_len_{}_{}_{}.npy'.format(ENV_NUM, STEP_N, lucky_no), ep_len_log)

        train_log = np.array(train_log)
        np.save('train_log_ERM_{}_{}_{}_{}.npy'.format(ENV_NUM, STEP_N, lucky_no, TRAIN_TIMES), train_log)

        temp_log = np.array(temp_log)
        np.save('temp_log_ERM_{}_{}_{}_{}.npy'.format(ENV_NUM, STEP_N, lucky_no, TRAIN_TIMES), temp_log)

    else:
        test(agent1, agent2, render=True, load_model=True)

    env.close()


if __name__ == '__main__':
    main()

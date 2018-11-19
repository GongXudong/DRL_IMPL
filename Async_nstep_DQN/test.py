from Env.cmotp_IL import CMOTP
from Async_nstep_DQN.lenient_DQNAgent_for_multi_envs import LenientDQNAgent
from Async_nstep_DQN.multi_env_IL_runner_wrapper import MultiEnvRunnerWrapper
import numpy as np
import time

DEBUG = False

TS_GREEDY_COEFF = 1.0

TRAIN = True
TRAIN = False
TEST_NUM = 10
TRAIN_NUM = 80000

GAMMA = 0.98

STEP_N = 5
ENV_NUM = 10


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


if __name__ == '__main__':
    env = MultiEnvRunnerWrapper(ENV_NUM, CMOTP)
    agent1 = LenientDQNAgent(env.envs[0], ENV_NUM, [256, 256], 'LenientAgent1',
                             learning_rate=1e-4,
                             use_tau=True, tau=1e-3,
                             logdir='logs1', savedir='save1', auto_save=False,
                             seed=25)

    agent2 = LenientDQNAgent(env.envs[0], ENV_NUM, [256, 256], 'LenientAgent2',
                             learning_rate=1e-4,
                             use_tau=True, tau=1e-3,
                             logdir='logs2', savedir='save2', auto_save=False,
                             seed=63)
    print('after init')

    if TRAIN:

        train_input_shape = (ENV_NUM * STEP_N,) + env.envs[0].observation_space.shape

        episodes_1 = [[] for _ in range(ENV_NUM)]
        episodes_2 = [[] for _ in range(ENV_NUM)]
        states_1, states_2 = env.reset()

        ep_cnt = 0

        ep_len_log = []
        min_len = 10000.

        for i in range(TRAIN_NUM):

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
                    ln_1[j].append(agent1.leniency_calculator.calc_leniency(agent1.temp_recorders[j].get_state_temp(states_1[j])))

                    sts_2[j].append(states_2[j])
                    acts_2[j].append(action_2)
                    rwds_2[j].append(reward_2)
                    n_sts_2[j].append(next_state_2)
                    dns_2[j].append(done_2)
                    ln_2[j].append(agent2.leniency_calculator.calc_leniency(agent2.temp_recorders[j].get_state_temp(states_2[j])))

                    states_1[j] = next_state_1
                    states_2[j] = next_state_2

                    if done_1:
                        states_1[j], states_2[j] = env.envs[j].reset()
                        agent1.temp_recorders[j].decay_temp(episodes_1[j])
                        agent2.temp_recorders[j].decay_temp(episodes_2[j])

                        ep_cnt += 1
                        print('train_num: {}, episode_cnt: {}, len: {} '.format(i, ep_cnt, len(episodes_1[j])))
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

    else:
        test_env = CMOTP()
        agent1.load_model()
        agent2.load_model()
        for i in range(TEST_NUM):
            state1, state2 = test_env.reset()
            while True:
                test_env.render()
                time.sleep(1)
                action1 = agent1.choose_action(state1, 0, 0.0)
                action2 = agent2.choose_action(state2, 0, 0.0)
                next_state, reward, done, _ = test_env.step([action1, action2])
                next_state1, next_state2 = next_state
                state1 = next_state1
                state2 = next_state2
                if done[0]:
                    break

    env.close()
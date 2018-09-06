
from Env.cmotp_JAL import CMOTP
from Async_nstep_DQN.multi_env_JAL_runner_wrapper import MultiEnvRunnerWrapper
from DQN.DQNAgent import DQNAgent
from Lenient.temperature_record import Temp_record
import time
import numpy as np

DEBUG = False

TS_GREEDY_COEFF = 1.0

# TRAIN = True
TRAIN = False
TEST_NUM = 10
TRAIN_NUM = 100000

GAMMA = 0.98

STEP_N = 5
ENV_NUM = 8

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

def main():
    env = MultiEnvRunnerWrapper(ENV_NUM, CMOTP)
    agent = DQNAgent(env.envs[0].observation_space.shape, env.envs[0].action_space.n, [512, 512], 'cmotp',
                     discount=GAMMA,
                     epsilon_decay_step=10000, epsilon_end=0.05, replay_memory_size=100000,
                     learning_rate=1e-4, targetnet_update_freq=5000, tau=1.)
    if TRAIN:
        temp_records = [Temp_record(shape=tuple(env.envs[0].observation_space.high + 1) + (env.envs[0].action_space.n,), beta_len=1500)
                        for _ in range(ENV_NUM)]

        train_input_shape = (ENV_NUM * STEP_N, ) + env.envs[0].observation_space.shape
        print(train_input_shape)

        episodes = [[] for _ in range(ENV_NUM)]
        states = env.reset()

        ep_cnt = 0

        for i in range(TRAIN_NUM):

            sts = [[] for _ in range(ENV_NUM)]
            acts = [[] for _ in range(ENV_NUM)]
            rwds = [[] for _ in range(ENV_NUM)]
            n_sts = [[] for _ in range(ENV_NUM)]
            dns = [[] for _ in range(ENV_NUM)]

            # get a batch of train data
            for j in range(ENV_NUM):
                for k in range(STEP_N):
                    action = agent.choose_action(states[j],
                                                 epsilon=pow(temp_records[j].get_state_temp(states[j]), TS_GREEDY_COEFF))
                    action_n = [int(action % 5), int(action / 5)]
                    n_st, rwd, dn, _ = env.envs[j].step(action_n)
                    # print(states[j], action, rwd, n_st, dn)
                    # record episodes
                    episodes[j].append((states[j], action))

                    # record train data
                    sts[j].append(states[j])
                    acts[j].append(action)
                    rwds[j].append(rwd)
                    n_sts[j].append(n_st)
                    dns[j].append(dn)

                    states[j] = n_st

                    if dn:
                        states[j] = env.envs[j].reset()
                        ep_cnt += 1
                        print('train_num: {}, episode_cnt: {}, len: {} '.format(i, ep_cnt, len(episodes[j])))
                        temp_records[j].decay_temp(episodes[j])
                        episodes[j] = []

            # discount reward
            last_values = agent.get_max_target_Q_s_a(states)
            for j, (rwd_j, dn_j, l_v_j) in enumerate(zip(rwds, dns, last_values)):
                if type(rwd_j) is np.ndarray:
                    rwd_j = rwd_j.tolist()
                if type(dn_j) is np.ndarray:
                    dn_j = dn_j.tolist()

                if dn_j[-1] == 0:
                    rwd_j = discount_with_dones(rwd_j + [l_v_j], dn_j + [0], GAMMA)[:-1]
                else:
                    rwd_j = discount_with_dones(rwd_j, dn_j, GAMMA)

                rwds[j] = rwd_j

            # flatten
            sts = np.asarray(sts, dtype=np.float32).reshape(train_input_shape)
            acts = np.asarray(acts, dtype=np.int32).flatten()
            rwds = np.asarray(rwds, dtype=np.float32).flatten()
            n_sts = np.asarray(n_sts, dtype=np.float32).reshape(train_input_shape)
            dns = np.asarray(dns, dtype=np.bool).flatten()

            # train
            agent.train_without_replaybuffer(sts, acts, rwds)

    else:
        # test
        test_env = CMOTP()
        agent.load_model()
        for i in range(TEST_NUM):
            state = test_env.reset()
            while True:
                test_env.render()
                time.sleep(1)
                action = agent.choose_action(state, epsilon=0.05)
                action_n = [int(action % 5), int(action / 5)]
                next_state, reward, done, _ = test_env.step(action_n)
                state = next_state
                if done:
                    break
        test_env.close()

    env.close()


if __name__ == '__main__':

    main()




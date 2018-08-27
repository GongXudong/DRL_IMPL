from Env.cmotp import CMOTP
from DQN.DQNAgent import DQNAgent
import numpy as np

EPISODES_NUM = 10000
FUTURE_SAMPLE_NUM = 100

TRAIN_NUM_PER_EPISODE = 100

MAX_EPISODE_LEN = 2000

TEST_NUM = 10

TRAIN = True

goals = np.array([np.array([2, 2, 2, 2, 4, 3]),
                  np.array([0, 2, 2, 0, 4, 3])])

def get_state(st: np.ndarray, gl: np.ndarray) -> np.ndarray:
    """

    :param st: np.array([1, 2])
    :param gl: np.array([3, 4])
    :return: np.array([1, 2, 3, 4])
    """
    return np.concatenate((st, gl), axis=0)


def get_reward_by_goal(st: np.ndarray, gl: np.ndarray) -> int:

    if np.all(st == gl):
        if np.all(gl == goals[0]):
            return 1
        if np.all(gl == goals[1]):
            return 10
        return 0
    return -1


if __name__ == '__main__':
    env = CMOTP()
    agent = DQNAgent(states_n=(env.observation_space.shape[0]*2, ),
                     actions_n=env.action_space.n,
                     hidden_layers=[256, 512],
                     scope_name='cmotp',
                     learning_rate=1e-4,
                     replay_memory_size=50000,
                     batch_size=32,
                     targetnet_update_freq=1000,
                     epsilon_end=0.05,
                     epsilon_decay_step=20000)

    episode_len = []

    if TRAIN:

        for episode_iter in range(EPISODES_NUM):
            print('in train_{}'.format(episode_iter))
            state = env.reset()
            goal = goals[1]
            len_this_episode = 0
            episode_record = []

            while True:
                action = agent.choose_action(get_state(state, goal))
                action_n = [int(action % 5), int(action / 5)]
                next_state, reward, done, _ = env.step(action_n)

                episode_record.append((state, action, reward, next_state, done, goal))
                agent.store(get_state(state, goal), action, reward, get_state(next_state, goal), done)

                len_this_episode += 1
                state = next_state

                if done or len_this_episode >= MAX_EPISODE_LEN:
                    break
            episode_len.append(len_this_episode)

            # hindsight
            for i in range(len_this_episode):
                st, ac, rw, nxt_st, dn, gl = episode_record[i]
                for k in range(min(FUTURE_SAMPLE_NUM, len_this_episode - i)):
                    idx = np.random.randint(i, len_this_episode)
                    _, _, _, nx_gl, _, _ = episode_record[idx]
                    agent.store(get_state(st, nx_gl), ac, get_reward_by_goal(st, nx_gl), get_state(nxt_st, nx_gl), dn)

                # train
            for i in range(TRAIN_NUM_PER_EPISODE):
                agent.train()

            if episode_iter % 100 == 99:
                print('episode {}, ave length: {}'.format((episode_iter + 1), np.mean(episode_len[-100:])))
    else:
        agent.load_model()

    # test
    for i in range(TEST_NUM):
        state = env.reset()
        goal = goals[1]
        len_this_episode = 0
        while True:
            action = agent.choose_action(get_state(state, goal), 0.02)
            next_state, reward, done, _ = env.step(action)
            print('action: ', action)
            print('next_state: ', next_state)
            state = next_state
            len_this_episode += 1
            if done:
                print('test_{}: action_num={}'.format(i, len_this_episode))
                break




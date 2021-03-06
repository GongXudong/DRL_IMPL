from Env.BitsGame import BitsGame
from DQN.DQNAgent import DQNAgent
import numpy as np

EPISODES_NUM = 10000
FUTURE_SAMPLE_NUM = 5

TRAIN_NUM_PER_EPISODE = 10

TEST_NUM = 10

TRAIN = False

def get_state(st: np.ndarray, gl: np.ndarray) -> np.ndarray:
    """

    :param st: np.array([1, 2])
    :param gl: np.array([3, 4])
    :return: np.array([1, 2, 3, 4])
    """
    return np.concatenate((st, gl), axis=0)


def get_reward_by_goal(st: np.ndarray, gl: np.ndarray) -> int:
    if np.all(st == gl):
        return 0
    return -1


if __name__ == '__main__':

    env = BitsGame(15)
    agent = DQNAgent(states_n=(env.size * 2, ),
                     actions_n=env.action_space.n,
                     hidden_layers=[256],
                     scope_name='BitsGame',
                     learning_rate=1e-4,
                     replay_memory_size=10000,
                     batch_size=32,
                     targetnet_update_freq=1000,
                     epsilon_end=0.05,
                     epsilon_decay_step=10000)

    if TRAIN:
        max_episode_len = env.observation_space.shape[0]
        rewards_record = []

        for episode_iter in range(EPISODES_NUM):
            state, goal = env.reset()
            reward_of_this_episode = 0
            len_of_this_episode = 0
            episode_record = []

            while True:

                action = agent.choose_action(get_state(state, goal))
                next_state, reward, done, _ = env.step(action)

                episode_record.append((state, action, reward, next_state, done, goal))
                agent.store(get_state(state, goal), action, reward, get_state(next_state, goal), done)

                len_of_this_episode += 1
                reward_of_this_episode += reward
                state = next_state

                if done or len_of_this_episode >= max_episode_len:
                    break

            rewards_record.append(reward_of_this_episode)

            # hindsight
            for i in range(len_of_this_episode):
                st, ac, rw, nxt_st, dn, gl = episode_record[i]
                for k in range(FUTURE_SAMPLE_NUM):
                    idx = np.random.randint(i, len_of_this_episode)
                    _, _, _, nx_gl, _, _ = episode_record[idx]
                    agent.store(get_state(st, nx_gl), ac, get_reward_by_goal(st, nx_gl), get_state(nxt_st, nx_gl), dn)

            # train
            for i in range(TRAIN_NUM_PER_EPISODE):
                agent.train()

            if episode_iter % 1000 == 999:
                print('episode {}, ave reward: {}'.format((episode_iter + 1), np.mean(rewards_record[-100:])))
    else:
        agent.load_model()

    # test
    for i in range(TEST_NUM):
        state, goal = env.reset()
        print('state: ', state)
        print('goal: ', goal)
        print('optima action_num:', np.sum(state != goal, axis=0))
        len_of_this_episode = 0
        while True:
            action = agent.choose_action(get_state(state, goal), 0.02)
            next_state, reward, done, _ = env.step(action)
            print('action: ', action)
            print('next_state: ', next_state)
            state = next_state
            len_of_this_episode += 1
            if done:
                print('actual action_num: ', len_of_this_episode)
                print()
                break
            if len_of_this_episode >= env.observation_space.shape[0]:
                print('failure!!!!')
                print()
                break


# from Env.cmotp_JAL import CMOTP
# import numpy as np
#
# if __name__ == '__main__':
#     env = CMOTP()
#
#     len_episodes = []
#
#     for i in range(10000):
#
#         state = env.reset()
#         env.map = [[0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, -1, 0, 0, 0],
#                     [0, 0, 1, 3, 2, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0],
#                     [-1, -1, 0, -1, -1, -1, -1],
#                     [0, 0, 0, 0, 0, 0, -1]]
#         len_this_episode = 0
#
#         while True:
#             action = env.action_space.sample()
#             action_n = [int(action % 5), int(action / 5)]
#             next_state, reward, done, _ = env.step(action_n)
#
#             len_this_episode += 1
#             state = next_state
#
#             if reward == 10.:
#                 break
#         print(i, len_this_episode)
#         len_episodes.append(len_this_episode)
#
#     print(np.mean(len_episodes))

from Env.cmotp_IL import CMOTP
import time
if __name__ == '__main__':
    env = CMOTP()
    state = env.reset()
    env.map = [[-1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, -1],
                    [0, 0, 0, 0, 2, 3, 1, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]]
    env.render()
    time.sleep(100)
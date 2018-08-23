import gym
import time
import numpy as np
import tensorflow as tf
from Hierarchical_DQN.H_DQNAgent1 import HierarchicalDQNAgent
from Hierarchical_DQN.Env.Stochastic_MDP import StochasticMDPEnv

# subgoals = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5])]
subgoals = [np.array([0]), np.array([5])]
subgoals_num = len(subgoals)


def one_hot(st, depth=6):
    vector = np.zeros(depth)
    if type(st) is np.ndarray:
        vector[st[0]] = 1.
    if type(st) is int:
        vector[st] = 1.
    return vector


def get_controller_state(st, gl):
    return np.concatenate((st, gl), axis=0)

if __name__ == '__main__':

    env = StochasticMDPEnv()


    INTRINSIC_REWARD = 1.
    INTRINSIC_STEP_COST = 0.

    check_subgoal_fn = lambda st, subgoal_index: np.all(st == subgoals[subgoal_index])

    agent = HierarchicalDQNAgent(original_states_n=(6, ),
                                 meta_controller_states_n=(6, ),
                                 actions_n=env.action_space.n,
                                 subgoals=subgoals,
                                 epsilon_decay_step=10000,
                                 epsilon_end=0.02)
    episode_lens = []
    episode_rewards = []

    for i in range(1000):
        state = env.reset()

        episode_len = 0
        episode_reward = 0
        while True:
            # iter for the episode

            # select a goal
            goal = agent.choose_goal(one_hot(state))

            state0 = state
            total_external_reward = 0.

            while True:
                # iter for a goal
                action = agent.choose_action(one_hot(state), one_hot(goal, subgoals_num))

                next_state, external_reward, done, _ = env.step(action)

                episode_reward += external_reward
                episode_len += 1

                goal_reached = check_subgoal_fn(next_state, goal)
                intrinsic_reward = INTRINSIC_REWARD if goal_reached else INTRINSIC_STEP_COST


                # Attention: the terminate state used here is goal_reached
                agent.controller.store(get_controller_state(one_hot(state), one_hot(goal, subgoals_num)),
                                       action, intrinsic_reward,
                                       get_controller_state(one_hot(next_state), one_hot(goal, subgoals_num)),
                                       goal_reached)

                agent.controller.train()
                agent.meta_controller.train()

                total_external_reward += external_reward
                state = next_state

                if goal_reached or done:
                    break

            # either the goal is reached or the terminal state is reached

            # for the meta_controller, the action is the goal
            # and the state transition for meta_controller is stochastic,
            # because for a goal_1, it may arrive the next_state or terminal_state

            agent.meta_controller.store(one_hot(state0), goal, total_external_reward, one_hot(next_state), done)

            if done:
                break

        # print('episode_{}: len {}, reward {}'.format(i, episode_len, episode_reward))
        episode_lens.append(episode_len)
        episode_rewards.append(episode_reward)
        if i % 1000 == 999:
            print('averaged lens: {}    averaged rewards: {}'.format(np.mean(episode_lens[-1000:]),
                                                                     np.mean(episode_rewards[-1000:])))
            time.sleep(2)

    agent.meta_controller.show_memory()
    env.close()









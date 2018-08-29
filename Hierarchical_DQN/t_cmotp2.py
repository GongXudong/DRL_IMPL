import time
import numpy as np
from Hierarchical_DQN.H_DQNAgent import HierarchicalDQNAgent
from Env.cmotp_JAL import CMOTP

subgoals = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5])]
# subgoals = [np.array([0]), np.array([5])]
subgoals_num = len(subgoals)

actor_epsilon = [1.] * subgoals_num
meta_epsilon = 1.
goal_select = [0] * subgoals_num
goal_success = [0] * subgoals_num
anneal_factor = (1. - 0.1)/12000

def check_subgoal_reached(st, subgoal_index):
    if subgoal_index == 0:
        return np.all(st == np.float32(np.array([2, 2, 2, 2, 4, 3])))
    if subgoal_index == 1:
        return

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

    env = CMOTP()


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

    for i in range(100000):
        state = env.reset()

        episode_len = 0
        episode_reward = 0
        while True:
            # iter for the episode

            # select a goal
            goal = agent.choose_goal(one_hot(state), epsilon=meta_epsilon)

            goal_select[goal] += 1

            state0 = state
            total_external_reward = 0.

            while True:
                # iter for a goal
                action = agent.choose_action(one_hot(state), one_hot(goal, subgoals_num),
                                             epsilon=actor_epsilon[goal])

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

                if goal_reached:
                    goal_success[goal] += 1

                if goal_reached or done:
                    break

            # either the goal is reached or the terminal state is reached

            # for the meta_controller, the action is the goal
            # and the state transition for meta_controller is stochastic,
            # because for a goal_1, it may arrive the next_state or terminal_state

            agent.meta_controller.store(one_hot(state0), goal, total_external_reward, one_hot(next_state), done)

            #Annealing
            meta_epsilon -= anneal_factor
            meta_epsilon = 0.1 if meta_epsilon < 0.1 else meta_epsilon

            actor_epsilon[goal] -= anneal_factor
            # avg_success_rate = goal_success[goal] / goal_select[goal]
            # if avg_success_rate == 0. or avg_success_rate == 1.:
            #     actor_epsilon[goal] -= anneal_factor
            # else:
            #     actor_epsilon[goal] = 1 - avg_success_rate
            actor_epsilon[goal] = 0.1 if actor_epsilon[goal] < 0.1 else actor_epsilon[goal]

            if done:
                break

        # print('episode_{}: len {}, reward {}'.format(i, episode_len, episode_reward))
        episode_lens.append(episode_len)
        episode_rewards.append(episode_reward)
        if i % 1000 == 999:
            print('episode:{}   averaged_lens: {}   averaged_rewards: {}'.format(int((i+1)/1000),
                                                                                 np.mean(episode_lens[-1000:]),
                                                                                 np.mean(episode_rewards[-1000:])))
            time.sleep(2)

    # agent.meta_controller.show_memory()
    env.close()









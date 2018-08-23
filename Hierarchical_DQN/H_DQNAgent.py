
import tensorflow as np
import numpy as np
import time
from Hierarchical_DQN.DQNAgent import DQNAgent

# refs: https://github.com/skumar9876/Hierarchical-DQN/blob/master/hierarchical_dqn.py
# 内部自动处理设置目标
class HierarchicalDQNAgent(object):

    INTRINSIC_STEP_COST = 0.
    INTRINSIC_REWARD = 1.
    def __init__(self,
                 original_states_n: tuple, meta_controller_states_n: tuple,
                 actions_n: int,
                 controller_hidden_layers=[32, 32, 32],
                 meta_controller_hidden_layers=[32, 32, 32],
                 discount=0.99,
                 controller_lr=0.1, meta_controller_lr=0.0001,
                 subgoals=None,
                 subgoals_mask=None,
                 meta_controller_state_fn=None,
                 check_subgoal_fn=None):
        """

        :param original_states_n: tuple
        :param meta_controller_states_n: tuple
        :param actions_n: int
        :param controller_lr:
        :param meta_controller_lr:
        :param subgoals: np.ndarray
        :param meta_controller_state_fn: lambda controller_state: meta_controller_state
        (a function that maps state for controller to another state for meta_controller)
        :param check_subgoal_fn: lambda state(np.ndarray), goal_num(int): bool
        (a function that checks whether the state achieves subgoals[goal_num])
        """

        self._subgoals = subgoals
        self._subgoals_mask = subgoals_mask
        self._num_subgoals = 0 if subgoals is None else len(subgoals)

        self._meta_controller = DQNAgent(states_n=meta_controller_states_n,
                                         actions_n=self._num_subgoals,
                                         hidden_layers=meta_controller_hidden_layers,
                                         scope_name='meta_controller',
                                         learning_rate=meta_controller_lr,
                                         epsilon_end=0.01,
                                         discount=discount)
        self._controller = DQNAgent(states_n=(original_states_n[0] + self._num_subgoals, ),
                                    actions_n=actions_n,
                                    hidden_layers=controller_hidden_layers,
                                    scope_name='controller',
                                    learning_rate=controller_lr,
                                    epsilon_end=0.01,
                                    discount=discount)



        self._meta_controller_state_fn = meta_controller_state_fn
        self._check_subgoal_fn = check_subgoal_fn

        # the following 5 variables are inner variables, there're no needing to process them from outside.
        self._meta_controller_state = None
        self._current_subgoal = None
        self._meta_controller_reward = 0
        self._intrinsic_time_step = 0
        self._episode = 0

    def show_current_goal(self):
        print(self._current_subgoal)

    def show_meta_controller_memory(self):
        self._meta_controller.show_memory()

    def get_meta_controller_state(self, state: tuple) -> np.ndarray:
        res = state
        if self._meta_controller_state_fn:
            res = self._meta_controller_state_fn(state)
        return np.array(res, copy=True)

    def get_controller_state(self, state: tuple, subgoal_index: int) -> np.ndarray:
        current_subgoal = np.array(self._subgoals_mask[subgoal_index])
        controller_state = np.array(state)
        controller_state = np.concatenate((controller_state, current_subgoal), axis=0)
        return controller_state

    def intrinsic_reward(self, state, subgoal_index) -> int:
        if self.subgoal_completed(state, subgoal_index):
            return self.INTRINSIC_REWARD
        else:
            return self.INTRINSIC_STEP_COST

    def subgoal_completed(self, state, subgoal_index) -> bool:
        if self._check_subgoal_fn is None:
            return state == self._subgoals[subgoal_index]
        else:
            return self._check_subgoal_fn(state, subgoal_index)

    def store(self, state, action, reward, next_state, terminate):
        intrinsic_state = self.get_controller_state(state, self._current_subgoal)
        intrinsic_next_state = self.get_controller_state(next_state, self._current_subgoal)
        intrinsic_reward = self.intrinsic_reward(next_state, self._current_subgoal)
        subgoal_completed = self.subgoal_completed(next_state, self._current_subgoal)

        # if next_state[0] == 6 or next_state[0] == 1:
        #     print('next_state: {}, goal: {}, finished: {}'.format(next_state[0], self._current_subgoal, subgoal_completed))
        #     time.sleep(2)

        intrinsic_terminate = terminate or subgoal_completed

        self._controller.store(intrinsic_state, action, intrinsic_reward, intrinsic_next_state, intrinsic_terminate)

        self._meta_controller_reward += reward

        if terminate:
            self._episode += 1

        if subgoal_completed:
            meta_controller_state = np.copy(self._meta_controller_state)
            meta_controller_next_state = self.get_meta_controller_state(next_state)

            self._meta_controller.store(meta_controller_state, self._current_subgoal,
                                        self._meta_controller_reward, meta_controller_next_state, terminate)

            # if meta_controller_state == np.array([6]):
            #     print(meta_controller_state, self._current_subgoal,
            #           self._meta_controller_reward, meta_controller_next_state, terminate)
            #     time.sleep(2)

        if subgoal_completed or terminate:
            self._meta_controller_state = None
            self._current_subgoal = None
            self._meta_controller_reward = 0
            self._intrinsic_time_step = 0

    def choose_action(self, state: np.ndarray) -> int:
        self._intrinsic_time_step += 1
        if self._current_subgoal is None:
            self._meta_controller_state = self.get_meta_controller_state(state)
            self._current_subgoal = self._meta_controller.choose_action(self._meta_controller_state)

        controller_state = self.get_controller_state(state, self._current_subgoal)
        action = self._controller.choose_action(controller_state)

        return action

    def check_meta_controller_output(self, state):
        self._meta_controller.check_network_output(state)

    def train(self):
        self._controller.train()
        if self._current_subgoal is None:
            self._meta_controller.train()


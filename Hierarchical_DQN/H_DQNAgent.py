
import tensorflow as np
import numpy as np
from DQN.DQNAgent import DQNAgent

# refs: https://github.com/skumar9876/Hierarchical-DQN/blob/master/hierarchical_dqn.py
# 设置目标等操作放在外面
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
                 subgoals_num=None,
                 epsilon_decay_step=10000,
                 epsilon_end=0.02):
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

        self._num_subgoals = subgoals_num

        self.meta_controller = DQNAgent(states_n=meta_controller_states_n,
                                        actions_n=self._num_subgoals,
                                        hidden_layers=meta_controller_hidden_layers,
                                        scope_name='meta_controller',
                                        learning_rate=meta_controller_lr,
                                        epsilon_decay_step=epsilon_decay_step,
                                        epsilon_end=epsilon_end,
                                        discount=discount)
        self.controller = DQNAgent(states_n=(original_states_n[0] + self._num_subgoals, ),
                                   actions_n=actions_n,
                                   hidden_layers=controller_hidden_layers,
                                   scope_name='controller',
                                   learning_rate=controller_lr,
                                   epsilon_decay_step=epsilon_decay_step,
                                   epsilon_end=epsilon_end,
                                   discount=discount)

    def choose_goal(self, state, epsilon=None):
        return self.meta_controller.choose_action(state, epsilon=epsilon)

    def choose_action(self, state, goal, epsilon=None):
        return self.controller.choose_action(np.concatenate((state, goal), axis=0), epsilon=epsilon)

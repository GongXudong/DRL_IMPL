
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import random
from common.schedules import LinearSchedule
from Hierarchical_DQN.replay_buffer import ReplayBuffer


class DQNAgent(object):
    """
    refs: https://github.com/skumar9876/Hierarchical-DQN/blob/master/dqn.py
    """
    def __init__(self, states_n: tuple, actions_n: int, hidden_layers: list, scope_name: str,
                 sess=None, learning_rate=0.001,
                 discount=0.98, replay_memory_size=100000, batch_size=32, begin_train=1000,
                 targetnet_update_freq=1000, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_step=50000,
                 seed=1, logdir='logs'):
        """

        :param states_n: tuple
        :param actions_n: int
        :param hidden_layers: list
        :param scope_name: str
        :param sess: tf.Session
        :param learning_rate: float
        :param discount: float
        :param replay_memory_size: int
        :param batch_size: int
        :param begin_train: int
        :param targetnet_update_freq: int
        :param epsilon_start: float
        :param epsilon_end: float
        :param epsilon_decay_step: int
        :param seed: int
        :param logdir: str
        """
        self.states_n = states_n
        self.actions_n = actions_n
        self._hidden_layers = hidden_layers
        self._scope_name = scope_name
        self.lr = learning_rate
        self._target_net_update_freq = targetnet_update_freq
        self._current_time_step = 0
        self._epsilon_schedule = LinearSchedule(epsilon_decay_step, epsilon_end, epsilon_start)
        self._train_batch_size = batch_size
        self._begin_train = begin_train
        self._gamma = discount

        self.qnet_optimizer = tf.train.AdamOptimizer(self.lr)

        self._replay_buffer = ReplayBuffer(replay_memory_size)

        self._seed(seed)

        with tf.Graph().as_default():
            self._build_graph()
            self._merged_summary = tf.summary.merge_all()
            self._saver = tf.train.Saver()
            if sess is None:
                self.sess = tf.Session()
            else:
                self.sess = sess
            self.sess.run(tf.global_variables_initializer())

            self._summary_writer = tf.summary.FileWriter(logdir=logdir)
            self._summary_writer.add_graph(tf.get_default_graph())

    def show_memory(self):
        print(self._replay_buffer.show())

    def _q_network(self, state, hidden_layers, outputs, scope_name, trainable):

        with tf.variable_scope(scope_name):
            out = state
            for ly in hidden_layers:
                out = layers.fully_connected(out, ly, activation_fn=tf.nn.relu, trainable=trainable)
            out = layers.fully_connected(out, outputs, activation_fn=None, trainable=trainable)
        return out

    def _build_graph(self):
        self._state = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.states_n, name='state_input')

        with tf.variable_scope(self._scope_name):
            self._q_values = self._q_network(self._state, self._hidden_layers, self.actions_n, 'q_network', True)
            self._target_q_values = self._q_network(self._state, self._hidden_layers, self.actions_n, 'target_q_network', False)

        with tf.variable_scope('q_network_update'):
            self._actions_onehot = tf.placeholder(dtype=tf.float32, shape=(None, self.actions_n), name='actions_onehot_input')
            self._td_targets = tf.placeholder(dtype=tf.float32, shape=(None, ), name='td_targets')
            self._q_values_pred = tf.reduce_sum(self._q_values * self._actions_onehot, axis=1)

            self._error = tf.abs(self._q_values_pred - self._td_targets)
            quadratic_part = tf.clip_by_value(self._error, 0.0, 1.0)
            linear_part = self._error - quadratic_part
            self._loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

            qnet_gradients = self.qnet_optimizer.compute_gradients(self._loss, tf.trainable_variables())
            for i, (grad, var) in enumerate(qnet_gradients):
                if grad is not None:
                    qnet_gradients[i] = (tf.clip_by_norm(grad, 10), var)
            self.train_op = self.qnet_optimizer.apply_gradients(qnet_gradients)

            tf.summary.scalar('loss', self._loss)


            with tf.name_scope('target_network_update'):
                q_network_params = [t for t in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                 scope=self._scope_name + '/q_network')
                                    if t.name.startswith(self._scope_name + '/q_network/')]
                target_q_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                            scope=self._scope_name + '/target_q_network')

                self.target_update_ops = []
                for var, var_target in zip(sorted(q_network_params, key=lambda v: v.name),
                                           sorted(target_q_network_params, key=lambda v: v.name)):
                    self.target_update_ops.append(var_target.assign(var))
                self.target_update_ops = tf.group(*self.target_update_ops)

    def choose_action(self, state, epsilon=None):

        if epsilon is not None:
            epsilon_used = epsilon
        else:
            epsilon_used = self._epsilon_schedule.value(self._current_time_step)
        if np.random.random() < epsilon_used:
            return np.random.randint(0, self.actions_n)
        else:
            q_values = self.sess.run(self._q_values, feed_dict={self._state: state[None]})

            return np.argmax(q_values[0])

    def check_network_output(self, state):
        q_values = self.sess.run(self._q_values, feed_dict={self._state: state[None]})
        print(q_values[0])

    def store(self, state, action, reward, next_state, terminate):
        self._replay_buffer.add(state, action, reward, next_state, terminate)

    def train(self):

        self._current_time_step += 1

        if self._current_time_step == 1:
            print('Training starts.')
            self.sess.run(self.target_update_ops)

        if self._current_time_step > self._begin_train:
            states, actions, rewards, next_states, terminates = self._replay_buffer.sample(batch_size=self._train_batch_size)

            actions_onehot = np.zeros((self._train_batch_size, self.actions_n))
            for i in range(self._train_batch_size):
                actions_onehot[i, actions[i]] = 1.

            next_state_q_values = self.sess.run(self._q_values, feed_dict={self._state: next_states})
            next_state_target_q_values = self.sess.run(self._target_q_values, feed_dict={self._state: next_states})

            next_select_actions = np.argmax(next_state_q_values, axis=1)
            next_select_actions_onehot = np.zeros((self._train_batch_size, self.actions_n))
            for i in range(self._train_batch_size):
                next_select_actions_onehot[i, next_select_actions[i]] = 1.

            next_state_max_q_values = np.sum(next_state_target_q_values * next_select_actions_onehot, axis=1)

            td_targets = rewards + self._gamma * next_state_max_q_values * (1 - terminates)

            _, str_ = self.sess.run([self.train_op, self._merged_summary], feed_dict={self._state: states,
                                                    self._actions_onehot: actions_onehot,
                                                    self._td_targets: td_targets})

            self._summary_writer.add_summary(str_, self._current_time_step)

        if self._current_time_step % self._target_net_update_freq == 0:
            self.sess.run(self.target_update_ops)

    def _seed(self, lucky_number):
        tf.set_random_seed(lucky_number)
        np.random.seed(lucky_number)
        random.seed(lucky_number)


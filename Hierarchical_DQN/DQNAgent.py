
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import random
from common.schedules import LinearSchedule
from Hierarchical_DQN.replay_buffer import ReplayBuffer
class DQNAgent(object):

    def __init__(self, states_n, actions_n, hidden_layers, scope_name, sess, learning_rate=0.001,
                 discount=0.98, replay_memory_size=100000, batch_size=32, rm_begin_work = 1000,
                 targetnet_update_freq = 1000, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_step=50000,
                 seed=1):
        self.states_n = states_n
        self.actions_n = actions_n
        self.lr = learning_rate

        self._current_time_step = 0
        self._epsilon_schedule = LinearSchedule(epsilon_decay_step, epsilon_end, epsilon_start)

        self._replay_buffer = ReplayBuffer(replay_memory_size)

        self._seed(seed)

        with tf.Graph().as_default():
            self._build_graph()
            self._saver = tf.train.Saver()
            if sess is None:
                self.sess = tf.Session()
            else:
                self.sess = sess
            self.sess.run(tf.global_variables_initializer())

    def _q_network(self, state, hidden_layers, scope_name, trainable):

        with tf.variable_scope(scope_name):
            out = state
            for ly in hidden_layers[:-1]:
                out = layers.fully_connected(out, ly, activation_fn=tf.nn.relu, trainable=trainable)
            out = layers.fully_connected(out, hidden_layers[-1], activation_fn=None, trainable=trainable)
        return out

    def _build_graph(self):
        self._state = tf.placeholder(dtype=tf.float32, shape=[None, self.states_n], name='state_input')

        self._q_values = self._q_network(self._state, 'q_network', True)
        self._target_q_values = self._q_network(self._state, 'target_q_network', False)

        with tf.variable_scope('q_network_update'):
            self._picked_actions = tf.placeholder(dtype=tf.float32, shape=[None, self.actions_n], name='actions_input')
            self._td_targets = tf.placeholder(dtype=tf.float32, shape=[None], name='td_targets')
            self._q_values_pred = tf.gather_nd(self._q_values, self._picked_actions)
            self._loss = tf.reduce_mean(self._clipped_error(self._q_values_pred - self._td_targets))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

            grads_and_vars = self.optimizer.compute_gradients(self._loss, tf.trainable_variables())

            grads = [gv[0] for gv in grads_and_vars]
            params = [gv[1] for gv in grads_and_vars]
            grads = tf.clip_by_global_norm(grads, 5.0)[0]

            clipped_grads_and_vars = zip(grads, params)

            self.train_op = self.optimizer.apply_gradients(clipped_grads_and_vars,
                                                           global_step=tf.contrib.framework.get_global_step())

            with tf.name_scope('target_network_update'):
                q_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_network')
                target_q_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_network')

                self.target_update_ops = []
                for var, var_target in zip(sorted(q_network_params, key=lambda v: v.name),
                                           sorted(target_q_network_params, key=lambda v: v.name)):
                    self.target_update_ops.append(var_target.assign(var))
                self.target_update_ops = tf.group(*self.target_update_ops)

    def choose_action(self, state):
        self._current_time_step += 1

        if np.random.random() < self._epsilon_schedule.value(self._current_time_step):
            return np.random.randint(0, self.actions_n)
        else:
            q_values = self.sess.run(self._q_values, feed_dict={self._state: state})
            return np.argmax(q_values)

    def store(self, state, action, reward, next_state, terminate):
        self._replay_buffer.add(state, action, reward, next_state, terminate)

    def train(self):
        pass

    def _clipped_error(self, x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

    def _seed(self, lucky_number):
        tf.set_random_seed(lucky_number)
        np.random.seed(lucky_number)
        random.seed(lucky_number)






"""
based on https://morvanzhou.github.io/tutorials/

Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.4
"""

import numpy as np
import tensorflow as tf
from Morvan.RL_brain import RL

np.random.seed(2)
tf.set_random_seed(2)


class PolicyGradient(RL):

    def __init__(self, *, n_actions, n_features, learning_rate=0.01, reward_decay=0.95, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.graph = self._build_graph()
        self.sess = self._build_session(self.graph)
        self.sess.run(self.init)
        self._reset_replay_buffer()
        if output_graph:
            tf.summary.FileWriter("F:/board/policy_gradient/", self.graph)

    def _reset_replay_buffer(self):
        self.rb_s = []
        self.rb_a = []
        self.rb_r = []  # 存储当前回报

    def _build_session(self, graph):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
        config.gpu_options.allow_growth = True  # 程序按需申请内存
        sess = tf.Session(graph=graph, config=config)
        return sess

    def _build_graph(self):
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            w_initializer = tf.random_normal_initializer(mean=0, stddev=0.3)
            b_initializer = tf.constant_initializer(value=0.1)

            with tf.name_scope("inputs"):
                self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="observations")
                self.a = tf.placeholder(dtype=tf.int32, shape=[None, ], name="actions")
                self.r = tf.placeholder(dtype=tf.float32, shape=[None, ], name="rewards")

            with tf.variable_scope("fc"):
                # [None, 10]
                hidden_layer = tf.layers.dense(inputs=self.s, units=10, activation=tf.nn.tanh,
                                               kernel_initializer=w_initializer,
                                               bias_initializer=b_initializer,
                                               name="fc1")

                # [None, n_actions]
                all_act = tf.layers.dense(inputs=hidden_layer, units=self.n_actions, activation=None,
                                          kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer,
                                          name="fc2")

                # [None, n_actions]
                self.all_act_prob = tf.nn.softmax(logits=all_act, name="act_prob")

            with tf.name_scope("loss"):
                # 求cross_entropy时自动乘了 -1
                # [None, ]
                neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.a)  # 为-log
                loss = tf.reduce_mean(neg_log_prob * self.r)  # reward在影响梯度方向

            with tf.name_scope("train"):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

            self.init = tf.global_variables_initializer()

        return graph

    def store_transition(self, s, a, r):
        self.rb_s.append(s)
        self.rb_a.append(a)
        self.rb_r.append(r)

    def _discount_and_normalize_reward_inplace(self):
        """
        将 self.rb_r从当期回报转变为未来回报 inplaced
        :return: None
        """
        length = len(self.rb_r)
        running_add = 0
        for i in range(length - 1, -1, -1):
            running_add = running_add * self.gamma + self.rb_r[i]  # 未来回报
            self.rb_r[i] = running_add

        mean = np.mean(self.rb_r)
        std = np.std(self.rb_r)

        for i in range(length):
            self.rb_r[i] = (self.rb_r[i] - mean) / (std + 1e-20)

    def choose_action(self, s):
        """
        select action based on observation s
        :param s: observation
        :return: action
        """
        s = np.asarray(s)[np.newaxis, :]  # reshape(1, -1)
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.s: s})

        # select randomly based on probability distribution calculated by softmax
        return np.random.choice(range(self.n_actions), p=prob_weights.ravel())

    def learn(self):
        self._discount_and_normalize_reward_inplace()
        self.sess.run(self.train_op, feed_dict={
            self.s: self.rb_s,
            self.a: self.rb_a,
            self.r: self.rb_r
        })

        vt = self.rb_r  # 暂存一下
        self._reset_replay_buffer()
        return vt


class ActorCritic(RL):
    def __init__(self, *, n_features, n_actions,
                 actor_learning_rate=0.001,
                 critic_learning_rate=0.01,
                 reward_decay=0.9,
                 output_graph=False):
        self.n_features = n_features
        self.n_actions = n_actions
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.gamma = reward_decay
        self.graph = self._build_graph()
        self.sess = self._build_session(self.graph)
        self.sess.run(self.init)
        if output_graph:
            tf.summary.FileWriter("F:/board/ActorCritic/", self.graph)

    def _build_session(self, graph):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
        config.gpu_options.allow_growth = True  # 程序按需申请内存
        sess = tf.Session(graph=graph, config=config)
        return sess

    def _build_graph(self):
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
            b_initializer = tf.constant_initializer(value=0.1)

            self.s = tf.placeholder(dtype=tf.float32, shape=[1, self.n_features], name="state")
            self.a = tf.placeholder(dtype=tf.int32, shape=None, name="act")
            self.v_ = tf.placeholder(dtype=tf.float32, shape=[1, 1], name="v_next")
            self.r = tf.placeholder(dtype=tf.float32, shape=None, name="reward")
            self.td_error_ = tf.placeholder(dtype=tf.float32, shape=None, name="td_error")

            with tf.variable_scope("Critic"):  # v函数模拟
                c_l1 = tf.layers.dense(inputs=self.s, units=20, activation=tf.nn.relu,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer,
                                       name="l1")
                # [1, 1]
                self.v = tf.layers.dense(inputs=c_l1, units=1, activation=None,
                                         kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer,
                                         name="V")

                with tf.variable_scope("td_loss"):
                    self.td_error = tf.reshape(self.r + self.gamma * self.v_ - self.v, [])
                    self.mse = tf.square(self.td_error)

                with tf.variable_scope("train"):
                    self.critic_train_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.mse)

            with tf.variable_scope("Actor"):
                a_l1 = tf.layers.dense(inputs=self.s, units=20, activation=tf.nn.relu,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer,
                                       name="l1")

                # [1, n_actions]
                self.acts_prob = tf.layers.dense(inputs=a_l1, units=self.n_actions, activation=tf.nn.softmax,
                                                 kernel_initializer=w_initializer,
                                                 bias_initializer=b_initializer,
                                                 name="acts_prob")

                with tf.variable_scope("action_loss"):
                    # shape: None
                    log_prob = tf.log(self.acts_prob[0, self.a])
                    self.log_loss = log_prob * self.td_error_

                with tf.variable_scope("train"):
                    self.actor_train_op = tf.train.AdamOptimizer(self.actor_lr).minimize(-self.log_loss)

            self.init = tf.global_variables_initializer()

        return graph

    def learn(self, s, a, r, s_):
        s = np.asarray(s)[np.newaxis, :]
        s_ = np.asarray(s_)[np.newaxis, :]
        v_ = self.sess.run(self.v, feed_dict={self.s: s_})
        # feed_dict = {self.s: s, self.a: a, self.v_: v_, self.r: r}
        td_error, _ = self.sess.run([self.td_error, self.critic_train_op],
                                    feed_dict={self.s: s, self.v_: v_, self.r: r})

        _, action_error = self.sess.run([self.actor_train_op, self.log_loss],
                                        feed_dict={self.s: s, self.td_error_: td_error, self.a: a})
        return td_error, action_error

    def choose_action(self, s):
        s = np.asarray(s)[np.newaxis, :]
        action_probs = self.sess.run(self.acts_prob, feed_dict={self.s: s})
        return np.random.choice(range(self.n_actions), p=action_probs.ravel())


# class ActorCritic(RL):
#     def __init__(self, *, n_features, n_actions,
#                  actor_learning_rate=0.001,
#                  critic_learning_rate=0.01,
#                  reward_decay=0.9,
#                  output_graph=False):
#         self.n_features = n_features
#         self.n_actions = n_actions
#         self.actor_lr = actor_learning_rate
#         self.critic_lr = critic_learning_rate
#         self.gamma = reward_decay
#         self._build_graph()
#         self.sess = self._build_session()
#         self.sess.run(self.init)
#         if output_graph:
#             tf.summary.FileWriter("F:/board/ActorCritic/", self.sess.graph)
#
#     def _build_session(self):
#         config = tf.ConfigProto()
#         config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
#         config.gpu_options.allow_growth = True  # 程序按需申请内存
#         sess = tf.Session(config=config, graph=self.graph)
#         return sess
#
#     def _build_graph(self):
#         tf.reset_default_graph()
#         self.graph = tf.Graph()
#         with self.graph.as_default():
#             self.actor = Actor(self.n_features, self.n_actions, self.actor_lr)
#             self.critic = Critic(self.n_features, self.gamma, self.critic_lr)
#             self.init = tf.global_variables_initializer()
#
#     def learn(self, s, a, r, s_):
#         s = np.asarray(s)[np.newaxis, :]
#         s_ = np.asarray(s_)[np.newaxis, :]
#         v_ = self.sess.run(self.critic.v, feed_dict={self.critic.s: s_})
#         td_error, _ = self.sess.run([self.critic.td_error, self.critic.train_op],
#                                     {self.critic.s: s, self.critic.v_: v_, self.critic.r: r})
#
#         feed_dict = {self.actor.s: s, self.actor.a: a, self.actor.td_error: td_error}
#         _, exp_v = self.sess.run([self.actor.train_op, self.actor.exp_v], feed_dict=feed_dict)
#
#
#     def choose_action(self, s):
#         s = np.asarray(s)[np.newaxis, :]
#         probs = self.sess.run(self.actor.acts_prob, {self.actor.s: s})  # get probabilities for all actions
#         return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int
#
#
# class Actor(object):
#     def __init__(self, n_features, n_actions, lr=0.001):
#         self.s = tf.placeholder(tf.float32, [1, n_features], "state")
#         self.a = tf.placeholder(tf.int32, None, "act")
#         self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
#
#         with tf.variable_scope('Actor'):
#             l1 = tf.layers.dense(
#                 inputs=self.s,
#                 units=20,    # number of hidden units
#                 activation=tf.nn.relu,
#                 kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='l1'
#             )
#
#             self.acts_prob = tf.layers.dense(
#                 inputs=l1,
#                 units=n_actions,    # output units
#                 activation=tf.nn.softmax,   # get action probabilities
#                 kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='acts_prob'
#             )
#
#         with tf.variable_scope('exp_v'):
#             log_prob = tf.log(self.acts_prob[0, self.a])
#             self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss
#
#         with tf.variable_scope('train'):
#             self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
#
#
# class Critic(object):
#     def __init__(self, n_features, gamma, lr=0.01):
#         self.s = tf.placeholder(tf.float32, [1, n_features], "state")
#         self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
#         self.r = tf.placeholder(tf.float32, None, 'r')
#         self.gamma = gamma
#
#         with tf.variable_scope('Critic'):
#             l1 = tf.layers.dense(
#                 inputs=self.s,
#                 units=20,  # number of hidden units
#                 activation=tf.nn.relu,  # None
#                 # have to be linear to make sure the convergence of actor.
#                 # But linear approximator seems hardly learns the correct Q.
#                 kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='l1'
#             )
#
#             self.v = tf.layers.dense(
#                 inputs=l1,
#                 units=1,  # output units
#                 activation=None,
#                 kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='V'
#             )
#
#         with tf.variable_scope('squared_TD_error'):
#             self.td_error = self.r + self.gamma * self.v_ - self.v
#             self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
#         with tf.variable_scope('train'):
#             self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
#
#

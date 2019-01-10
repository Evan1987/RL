"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetWork:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.graph = self.graph()
        self.sess = self.session()
        self.learning_step_count = 0

        # feature_num: [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

    @property
    def session(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
        config.gpu_options.allow_growth = True  # 程序按需申请内存
        sess = tf.Session(graph=self.graph, config=config)
        return sess

    @property
    def graph(self):
        """构件图"""
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():

            t_params = tf.get_collection("target_net_params")
            e_params = tf.get_collection("eval_net_params")
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

            # ------------------ build evaluate_net ------------------
            self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="s")
            self.q_target = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name="Q_target")
            with tf.variable_scope("eval_net"):
                c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
                n_l1 = 10
                w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.3)
                b_initializer = tf.constant_initializer(0.1, dtype=tf.float32)

                # first layer. collections is used later when assign to target net
                with tf.variable_scope("l1"):
                    w1 = tf.get_variable(name="w1", shape=[self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable(name="b1", shape=[1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(self.s @ w1 + b1)

                # second layer. collections is used later when assign to target net
                with tf.variable_scope("l2"):
                    w2 = tf.get_variable(name="w2", shape=[n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable(name="b2", shape=[1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.q_eval = l1 @ w2 + b2

            with tf.variable_scope("loss"):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            with tf.variable_scope("train"):
                self.train_op = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)

            # ------------------ build target_net ------------------
            self.s_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="s_")
            with tf.variable_scope("target_net"):
                # c_names(collections_names) are the collections to store variables
                c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

                with tf.variable_scope("l1"):
                    w1 = tf.get_variable(name="w1", shape=[self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable(name="b1", shape=[1, n_l1], initializer=w_initializer, collections=c_names)
                    l1 = tf.nn.relu(self.s_ @ w1 + b1)

                with tf.variable_scope("l2"):
                    w2 = tf.get_variable(name="w2", shape=[n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable(name="b2", shape=[1, self.n_actions], initializer=w_initializer, collections=c_names)
                    self.q_next = l1 @ w2 + b2

        return graph

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, "memory_counter"):
            self.memory_counter = 0

        transition = np.hstack((s, a, r, s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]

        rand = np.random.uniform()
        if rand < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: s})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(low=0, high=self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learning_step_count % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print("Target params replaced!")

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(np.arange(self.memory_size), size=self.batch_size)
        else:
            sample_index = np.random.choice(np.arange(self.memory_counter), size=self.batch_size)

        batch_memory = self.memory[sample_index, :]  # column: [---s---, a, r, ---s_---], ---s---: length(self.n_features)
        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
                                       feed_dict={self.s: batch_memory[:, :self.n_features],
                                                  self.s_: batch_memory[:, -self.n_features:]})
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=tf.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)  # a
        reward = batch_memory[:, self.n_features + 1]  # r

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)


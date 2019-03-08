"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
"""

import numpy as np
import matplotlib.pyplot as plt
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
        self.replace_target_iter = replace_target_iter  # 每隔**步同步两个网络的参数
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.sess = self.session
        self.sess.run(self.init)
        self.learning_step_count = 0

        # feature_num: [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        self.cost_list = []

        if output_graph:
            tf.summary.FileWriter("F:/board/DQN/", self.graph)

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

            e_params = tf.get_collection("eval_net_params")
            t_params = tf.get_collection("target_net_params")
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

            # ------------------ all inputs --------------------------
            self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="s")
            self.s_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="s_")
            self.r = tf.placeholder(dtype=tf.float32, shape=[None, ], name="r")
            self.a = tf.placeholder(dtype=tf.int32, shape=[None, ], name="a")
            w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.3)
            b_initializer = tf.constant_initializer(0.1)

            # ------------------ build evaluate_net ------------------
            with tf.variable_scope("eval_net"):
                # shape: [None, 20]  status => hidden_layer
                e1 = tf.layers.dense(inputs=self.s, units=20, activation=tf.nn.relu,
                                     kernel_initializer=w_initializer, bias_initializer=b_initializer, name="e1")

                # shape: [None, n_actions]   hidden_layer => action
                self.q_eval = tf.layers.dense(inputs=e1, units=self.n_actions, activation=None,
                                              kernel_initializer=w_initializer, bias_initializer=b_initializer, name="q")

            # ------------------ build target_net ------------------
            with tf.variable_scope("target_net"):
                # shape: [None, 20]
                t1 = tf.layers.dense(inputs=self.s_, units=20, activation=tf.nn.relu,
                                     kernel_initializer=w_initializer, bias_initializer=b_initializer, name="t1")

                # shape: [None, n_actions]
                self.q_next = tf.layers.dense(inputs=t1, units=self.n_actions, activation=None,
                                              kernel_initializer=w_initializer, bias_initializer=b_initializer,
                                              name="t2")

            with tf.variable_scope("q_eval"):  # Pred Q
                a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)  # [None, 2]
                self.q_eval_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # [None, ]  get new tensor from params by indices

            with tf.variable_scope("q_target"):  # Actual Q simulated by Q learning
                # shape: [None, ]
                q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name="Qmax_s_")
                self.q_target = tf.stop_gradient(q_target)  # don't take its value into account when computing gradient

            with tf.variable_scope("loss"):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_a, name="TD_error"))

            with tf.variable_scope("train"):
                self.train_op = tf.train.RMSPropOptimizer(self.alpha).minimize(self.loss)

            self.init = tf.global_variables_initializer()

        return graph

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, "memory_counter"):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

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

        _, cost = self.sess.run([self.train_op, self.loss], feed_dict={
            self.s: batch_memory[:, :self.n_features],
            self.a: batch_memory[:, self.n_features],
            self.r: batch_memory[:, self.n_features + 1],
            self.s_: batch_memory[:, -self.n_features:]
        })

        self.cost_list.append(cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learning_step_count += 1

        if self.learning_step_count > 0 and self.learning_step_count % 50 == 0:
            print("Iterations: %d  cost: %.4f" % (self.learning_step_count, cost))
            print(self.memory)

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_list)), self.cost_list)
        plt.ylabel("Cost")
        plt.xlabel("training_steps")
        plt.show()


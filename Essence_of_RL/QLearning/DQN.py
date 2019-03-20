
"""
The DQN improvement: Prioritized Experience Replay (based on https://morvanzhou.github.io/tutorials/)

Using:
Tensorflow: 1.4
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity

        # tree:
        # Store p(not normalized) of parents nodes and leaf nodes
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.tree = np.zeros(2 * capacity - 1)  # 长度为奇数，末端索引为偶数
        self.data = np.zeros(capacity, dtype=object)  # 这样可保证存储任何数据

    def add(self, data, p):
        """
        增加样本，移动 data_pointer，更新权重
        :param data: 待存储数据
        :param p: 数据的权重
        :return: None
        """
        # tree_idx 与 data_idx 的转换关系
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # 重置写入指针，如果写入已经exceed
            self.data_pointer = 0

    def update(self, tree_idx, p):
        """
        更新线段树的节点权重
        :param tree_idx: 目标节点
        :param p: 新权重
        :return: None
        """
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        # 更新叶子节点所属的各级父节点
        while tree_idx != 0:
            # 子节点与直接父节点从属关系
            # (children_index - 1) // 2 -> parent_index
            # parent_index -> (children_left: parent_index * 2 + 1, children_right: parent_index * 2 + 2)
            # 左子节点索引一定为奇数，右子节点索引一定为偶数
            # // 2确保了最多只有两个子节点从属同一父节点
            # -1确保了最后一个叶子节点可以归属到父节点，叶子节点index范围 [N-1, 2N-2]
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        根据输入的 v，选择唯一的样本输出
        :param v:  输入的随机数
        :return: tuple, 样本的树索引，样本权重，样本数据
        """
        parent_idx = 0
        while True:
            children_left = 2 * parent_idx + 1
            children_right = children_left + 1
            if children_left >= len(self.tree):  # 索引触底
                leaf_idx = parent_idx
                break
            if v <= self.tree[children_left]:
                parent_idx = children_left
            else:
                v -= self.tree[children_left]
                parent_idx = children_right

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self) -> float:
        return self.tree[0]


class Memory(object):

    def __init__(self, capacity, *, alpha=0.6, beta=0.4,
                 beta_increment_per_sampling=0.001, abs_err_upper=1, epsilon=0.01):
        """
        Priority Replay Buffer数据结构
        :param capacity: 容量 N
        :param alpha: 采样权重调节因子
        :param beta: 样本权重调节因子
        :param beta_increment_per_sampling: 样本权重调节因子的增长控制因子
        :param abs_err_upper: clip Td-error的上限值
        :param epsilon: Td-error辅助增值，to avoid 0
        """
        self.capacity = capacity
        self.sumTree = SumTree(capacity)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.abs_err_upper = abs_err_upper
        self.epsilon = epsilon

    def _get_priority(self, td_errors: np.ndarray):
        """
        由 td_error到采样权重的转换公式
        :param td_errors:
        :return:
        """
        return np.power(np.minimum(np.abs(td_errors) + self.epsilon, self.abs_err_upper), self.alpha)

    def store(self, sample):
        """
        储存数据。只是单纯存储，还未计算td_error，因此权重为默认值（取当前最大）
        :param sample: 待储存样本
        :return: None
        """
        max_p = np.max(self.sumTree.tree[-self.capacity:])
        max_p = self.abs_err_upper if max_p == 0 else max_p
        self.sumTree.add(sample, max_p)

    def sample(self, n: int) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        从存储区采样
        :param n: 样本数量
        :return: 各样本的树索引， 样本权重列表， 样本数据
        """
        batch_idx = np.empty(shape=(n, ), dtype=np.int32)
        batch_weights = np.empty(shape=(n, 1), dtype=np.float32)

        data_size = self.sumTree.data[0].size
        assert data_size == 6
        batch_memory = np.empty(shape=(n, data_size))
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)  # clip beta

        segment = self.sumTree.total_p / n  # 分成区间分别采样，保证随机数均衡性，依然保证了均匀随机
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            v = random.uniform(a, b)
            idx, p, data = self.sumTree.get_leaf(v)
            batch_idx[i] = idx
            batch_weights[i, 0] = (p / self.sumTree.total_p) ** -self.beta
            batch_memory[i, :] = data
        return batch_idx, batch_weights, batch_memory

    def batch_update(self, tree_idx: list, abs_errors: np.ndarray):
        """
        更新样本权重
        :param tree_idx: 待更新样本索引列表
        :param abs_errors: 待更新样本的新 abs(td_error)
        :return: None
        """
        for idx, p in zip(tree_idx, self._get_priority(abs_errors)):
            self.sumTree.update(idx, p)


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
                 double_q=False,
                 prioritized=True,
                 dueling=False,
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
        self.double_q = double_q  # 是否使用doubleDQN
        self.prioritized = prioritized  # 是否使用PRB
        self.dueling = dueling  # 是否使用Dueling
        self.graph = self.build_graph()
        self.sess = self.build_session(self.graph)
        self.sess.run(self.init)
        self.learning_step_count = 0

        if prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            # feature_num: [s, a, r, s_]
            self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        self.cost_list = []

        if output_graph:
            tf.summary.FileWriter("F:/board/DQN/", self.graph)

    def build_session(self, graph):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
        config.gpu_options.allow_growth = True  # 程序按需申请内存
        sess = tf.Session(graph=graph, config=config)
        return sess

    def build_layers(self, s, w_initializer, b_initializer, hidden_units):
        e1 = tf.layers.dense(inputs=s, units=hidden_units, activation=tf.nn.relu,
                             kernel_initializer=w_initializer,
                             bias_initializer=b_initializer,
                             name="e1")
        if self.dueling:
            with tf.variable_scope("Value"):
                self.V = tf.layers.dense(inputs=e1, units=1, activation=None,
                                         kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer,
                                         name="v")
            with tf.variable_scope("Advantage"):
                # [None, n_actions]
                self.A = tf.layers.dense(inputs=e1, units=self.n_actions, activation=None,
                                         kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer,
                                         name="a")
            with tf.variable_scope("Q"):
                out = self.V + self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True)
        else:
            with tf.variable_scope("Q"):
                out = tf.layers.dense(inputs=e1, units=self.n_actions, activation=None,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name="q")
        return out

    def build_graph(self):
        """构建图"""
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")
            with tf.variable_scope("hard_replacement"):
                self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

            # ------------------ all inputs --------------------------
            self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="s")
            self.s_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="s_")
            self.r = tf.placeholder(dtype=tf.float32, shape=[None, ], name="r")
            self.a = tf.placeholder(dtype=tf.int32, shape=[None, ], name="a")
            if self.prioritized:
                self.weights = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="weights")
            w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.3)
            b_initializer = tf.constant_initializer(0.1)

            # ------------------ build evaluate_net ------------------
            with tf.variable_scope("eval_net"):
                self.q_eval = self.build_layers(self.s, w_initializer, b_initializer, 20)

            # ------------------ build target_net ------------------
            with tf.variable_scope("target_net"):
                self.q_next = self.build_layers(self.s_, w_initializer, b_initializer, 20)

            with tf.variable_scope("q_eval"):  # Pred Q
                a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)  # [None, 2]
                self.q_eval_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # [None, ]

            with tf.variable_scope("q_target"):  # Actual Q simulated by Q learning
                # shape: [None, ]
                if self.double_q:
                    indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32),
                                        tf.argmax(self.q_eval, axis=1, output_type=tf.int32)], axis=1)
                    selected_q_next = tf.gather_nd(params=self.q_next, indices=indices)  # [None, ]
                    # 对于Double DQN q_target的估计需要依赖eval网络，所以需要计算梯度
                    self.q_target = self.r + self.gamma * selected_q_next  # [None, ]
                else:
                    selected_q_next = tf.reduce_max(self.q_next, axis=1, name="Qmax_s_")
                    q_target = self.r + self.gamma * selected_q_next
                    # 对于一般DQN，q_target来源于target网络，因此不需要计算梯度
                    self.q_target = tf.stop_gradient(q_target)  # [None, ]

            with tf.variable_scope("loss"):
                if self.prioritized:
                    self.abs_errors = tf.abs(self.q_target - self.q_eval_a)
                    self.loss = tf.reduce_mean(self.weights * tf.squared_difference(self.q_target, self.q_eval_a))
                else:
                    self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_a))

            with tf.variable_scope("train"):
                self.train_op = tf.train.RMSPropOptimizer(self.alpha).minimize(self.loss)

            self.init = tf.global_variables_initializer()

        return graph

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))

        if self.prioritized:
            self.memory.store(transition)

        else:
            if not hasattr(self, "memory_counter"):
                self.memory_counter = 0

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

        if self.prioritized:
            tree_idx, weights, batch_memory = self.memory.sample(self.batch_size)
            _, abs_errors, cost = self.sess.run([self.train_op, self.abs_errors, self.loss], feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
                self.weights: weights
            })

            # 随着训练进行， 样本的td_error会发生改变，及时更新
            self.memory.batch_update(tree_idx, abs_errors)
        else:
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
            if not self.prioritized:
                print(self.memory)

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_list)), self.cost_list)
        plt.ylabel("Cost")
        plt.xlabel("training_steps")
        plt.show()


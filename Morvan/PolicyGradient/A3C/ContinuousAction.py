"""
Based on https://morvanzhou.github.io/tutorials/
"""

import gym
import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
tf.set_random_seed(2)
GAME = 'Pendulum-v0'
UPDATE_GLOBAL_ITER = 10
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
MAX_EP_STEP = 200  # Pendulum never terminated, so set a max_loop
MAX_GLOBAL_EP = 2000
N_WORKERS = multiprocessing.cpu_count()
OUTPUT_GRAPH = True

#
# def _build_session(graph):
#     config = tf.ConfigProto()
#     config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
#     config.gpu_options.allow_growth = True  # 程序按需申请内存
#     sess = tf.Session(graph=graph, config=config)
#     return sess


class ACNet(object):
    def __init__(self, *, scope: str, n_actions: int, n_features: int, action_bounds: list, reward_decay=0.9,
                 entropy_beta=0.01, actor_learning_rate=0.0001, critic_learning_rate=0.001,
                 is_global: bool=False, globalAC=None):
        self.scope = scope
        self.n_actions = n_actions  # 1
        self.n_features = n_features
        self.action_low, self.action_high = action_bounds
        self.gamma = reward_decay
        self.entropy_beta = entropy_beta
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.is_global = is_global
        self.globalAC = globalAC
        self._build_graph()

    def _build_net(self):
        with tf.variable_scope("critic"):
            c_l1 = tf.layers.dense(inputs=self.s, units=100, activation=tf.nn.relu6,
                                   kernel_initializer=self.w_init,
                                   bias_initializer=self.b_init,
                                   name="c_l1")
            v = tf.layers.dense(inputs=c_l1, units=1, activation=None,
                                kernel_initializer=self.w_init,
                                bias_initializer=self.b_init,
                                name="V")

        with tf.variable_scope("actor"):
            a_l1 = tf.layers.dense(inputs=self.s, units=200, activation=tf.nn.relu6,
                                   kernel_initializer=self.w_init,
                                   bias_initializer=self.b_init,
                                   name="a_l1")
            # [None, n_actions]
            mu = tf.layers.dense(inputs=a_l1, units=self.n_actions, activation=tf.nn.tanh,
                                 kernel_initializer=self.w_init,
                                 bias_initializer=self.b_init,
                                 name="mu")
            # [None, n_actions]
            sigma = tf.layers.dense(inputs=a_l1, units=self.n_actions, activation=tf.nn.softplus,
                                    kernel_initializer=self.w_init,
                                    bias_initializer=self.b_init,
                                    name="sigma")

        actor_params = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope=self.scope + "/actor")
        critic_params = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope=self.scope + "/critic")
        return mu, sigma, v, actor_params, critic_params

    def _build_graph(self):
        self.w_init = tf.random_normal_initializer(mean=0, stddev=0.1)
        self.b_init = tf.constant_initializer(0.0)
        if self.is_global:
            with tf.variable_scope(self.scope):
                self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="S")
                _, _, _, self.actor_params, self.critic_params = self._build_net()
        else:
            with tf.variable_scope(self.scope):
                self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="S")
                self.a_his = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name="A")  # 分布中取概率密度，需要float32
                self.v_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="V_target")
                mu, sigma, self.v, self.actor_params, self.critic_params = self._build_net()

                td = tf.subtract(self.v_target, self.v, name="td_error")  # tf.subtract element-wise x - y
                with tf.name_scope("critic_loss"):
                    # mse tf.reduce_mean(tf.squared_difference(x, y))
                    critic_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope("wrap_action_out"):
                    mu = mu * self.action_high
                    sigma = sigma + 1e-4

                normal_dist = tf.distributions.Normal(loc=mu, scale=sigma)  # action的概率分布

                with tf.name_scope("actor_loss"):
                    log_prob = normal_dist.log_prob(value=self.a_his)
                    weighted_log_prob = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()
                    actor_loss = tf.reduce_mean(-(entropy * self.entropy_beta + weighted_log_prob))

                with tf.name_scope("choose_action"):
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(sample_shape=1), axis=[0, 1]),
                                              clip_value_min=self.action_low,
                                              clip_value_max=self.action_high)

                with tf.name_scope("local_grad"):
                    self.actor_grads = tf.gradients(actor_loss, self.actor_params)
                    self.critic_grads = tf.gradients(critic_loss, self.critic_params)

                with tf.name_scope("opt"):
                    actor_opt = tf.train.RMSPropOptimizer(learning_rate=self.actor_lr)
                    critic_opt = tf.train.RMSPropOptimizer(learning_rate=self.critic_lr)

            with tf.name_scope("sync"):
                with tf.name_scope("pull"):
                    """从global同步参数"""
                    self.actor_params_pull_op = [local_p.assign(global_p) for local_p, global_p in
                                                 zip(self.actor_params, self.globalAC.actor_params)]
                    self.critic_params_pull_op = [local_p.assign(global_p) for local_p, global_p in
                                                  zip(self.critic_params, self.globalAC.critic_params)]
                with tf.name_scope("push"):
                    """将梯度更新至global"""
                    self.actor_params_push_op = \
                        actor_opt.apply_gradients(zip(self.actor_grads, self.globalAC.actor_params))
                    self.critic_params_push_op = \
                        critic_opt.apply_gradients(zip(self.critic_grads, self.globalAC.critic_params))

    def update_global(self, *, sess, s, a, v_target):
        s = np.asarray(s).reshape(-1, self.n_features)
        a = np.asarray(a).reshape(-1, self.n_actions)
        v_target = np.asarray(v_target).reshape(-1, 1)
        feed_dict = {self.s: s, self.a_his: a, self.v_target: v_target}
        sess.run([self.actor_params_push_op, self.critic_params_push_op], feed_dict=feed_dict)

    def sync_with_global(self, sess):
        sess.run([self.actor_params_pull_op, self.critic_params_pull_op])

    def choose_action(self, sess: tf.Session, s):
        s = np.asarray(s).reshape(-1, self.n_features)
        return sess.run(self.A, feed_dict={self.s: s})


class Worker(object):
    def __init__(self, scope, globalAC: ACNet):
        self.scope = scope
        self.env: gym.Env = gym.make(GAME).unwrapped
        self.globalAC = globalAC
        self.AC = ACNet(scope=scope,
                        n_actions=self.env.action_space.shape[0],
                        n_features=self.env.observation_space.shape[0],
                        action_bounds=[self.env.action_space.low[0], self.env.action_space.high[0]],
                        is_global=False,
                        globalAC=globalAC)
        self._reset_buffer()

    def _reset_buffer(self):
        self.buffer_s = []
        self.buffer_a = []
        self.buffer_r = []

    def work(self, COORD, SESS):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0

            for ep_t in range(MAX_EP_STEP):
                a = self.AC.choose_action(sess=SESS, s=s)
                s_, r, done, info = self.env.step(a)  # need a (1,) shape
                done = True if ep_t == MAX_EP_STEP - 1 else False  # 最后一个循环为done
                ep_r += r
                self.buffer_s.append(s)
                self.buffer_a.append(a)
                self.buffer_r.append((r + 8) / 8)  # normalized to (-1, 1)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    v_s_ = 0 if done else np.asscalar(SESS.run(self.AC.v, feed_dict={self.AC.s: [s_]}))
                    # trans buffer_r to buffer_v_target by v(s) = gamma * v(s_) + r circularly
                    for i in range(len(self.buffer_r) - 1, -1, -1):
                        v_s_ = v_s_ * self.AC.gamma + self.buffer_r[i]
                        self.buffer_r[i] = v_s_
                    # maybe misleading buffer_r is buffer_v_target now
                    self.AC.update_global(sess=SESS, s=self.buffer_s, a=self.buffer_a, v_target=self.buffer_r)  # 更新
                    self.AC.sync_with_global(sess=SESS)
                    self._reset_buffer()
                s = s_
                total_step += 1

                if done:
                    if not GLOBAL_RUNNING_R:
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print("%s  E.P.: %d  EP_r: %i" % (self.scope, GLOBAL_EP, GLOBAL_RUNNING_R[-1]))
                    GLOBAL_EP += 1
                    break


if __name__ == "__main__":
    env = gym.make(GAME).unwrapped
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]
    n_features = env.observation_space.shape[0]

    tf.reset_default_graph()
    graph = tf.Graph()
    with tf.device("/CPU:0"):  # 都放在内存里
        with graph.as_default():
            GLOBAL_AC = ACNet(scope="GLOBAL", n_actions=n_actions, action_bounds=action_bounds,
                              n_features=n_features, is_global=True)
            workers = []
            for i in range(N_WORKERS):
                scope = "Worker_%d" % i
                workers.append(Worker(scope=scope, globalAC=GLOBAL_AC))
            init = tf.global_variables_initializer()

    SESS = tf.Session(graph=graph, config=tf.ConfigProto(device_count={'GPU': 0}))  # 使用cpu计算
    SESS.run(init)
    COORD = tf.train.Coordinator()

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("F:/board/A3C/", graph)

    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=lambda: worker.work(COORD=COORD, SESS=SESS))
        t.start()
        worker_threads.append(t)
    COORD.join(threads=worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel("step")
    plt.ylabel("Total moving reward")
    plt.show()
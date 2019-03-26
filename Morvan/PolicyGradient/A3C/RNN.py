"""
Asynchronous Advantage Actor Critic (A3C) + RNN with continuous action space, Reinforcement Learning.

The Pendulum example.

Based on: https://morvanzhou.github.io/tutorials/
"""

import multiprocessing
import threading
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell
import numpy as np
import gym
import matplotlib.pyplot as plt

GAME = "Pendulum-v0"
OUTPUT_GRAPH = True
MAX_EP_STEP = 200
MAX_GLOBAL_EP = 1500
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

env: gym.Env = gym.make(GAME).unwrapped
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]


class ACNet(object):
    cell_size = 64
    gamma = GAMMA
    actor_lr = LR_A
    critic_lr = LR_C
    entropy_beta = ENTROPY_BETA
    n_actions = N_A
    n_features = N_S
    action_low, action_high = A_BOUND
    sess = None  # reassigned from outer lately

    def __init__(self, scope: str, is_global: bool, globalAC=None):
        self.scope = scope
        self.is_global = is_global
        if globalAC is not None:
            self.globalAC = globalAC

        self._build_graph()

    def _build_net(self):
        with tf.variable_scope("critic"):
            with tf.variable_scope("state_input"):  # 这里仅让input与critic相关
                # add dim, [time_step, feature] => [time_step, batch_size=1, feature]
                s = tf.expand_dims(input=self.s, axis=1, name="timely_input")
                rnn_cell = BasicRNNCell(self.cell_size)
                self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)

                # output: [time_step, batch_size, cell_size]
                # final_state: [batch_size, cell_size]
                output, self.final_state = tf.nn.dynamic_rnn(
                    cell=rnn_cell, inputs=s, initial_state=self.init_state, time_major=True, dtype=tf.float32
                )
                cell_out = tf.reshape(tensor=output, shape=[-1, self.cell_size], name="flatten_rnn_outputs")
            lc = tf.layers.dense(inputs=cell_out, units=50, activation=tf.nn.relu6,
                                 kernel_initializer=self.w_init,
                                 bias_initializer=self.b_init,
                                 name="lc")
            v = tf.layers.dense(inputs=lc, units=1, activation=None,
                                kernel_initializer=self.w_init,
                                bias_initializer=self.b_init,
                                name="V")

        with tf.variable_scope("actor"):
            la = tf.layers.dense(inputs=cell_out, units=80, activation=tf.nn.relu6,
                                 kernel_initializer=self.w_init,
                                 bias_initializer=self.b_init,
                                 name="la")
            mu = tf.layers.dense(inputs=la, units=self.n_actions, activation=tf.nn.tanh,
                                 kernel_initializer=self.w_init,
                                 bias_initializer=self.b_init,
                                 name="mu")
            sigma = tf.layers.dense(inputs=la, units=self.n_actions, activation=tf.nn.softplus,
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
                self.a_his = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name="A")
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

    def update_global(self, *, s, a, v_target, cell_state):
        s = np.asarray(s).reshape(-1, self.n_features)
        a = np.asarray(a).reshape(-1, self.n_actions)
        v_target = np.asarray(v_target).reshape(-1, 1)
        cell_state = np.asarray(cell_state).reshape(-1, self.cell_size)
        feed_dict = {self.s: s, self.a_his: a, self.v_target: v_target, self.init_state: cell_state}
        self.sess.run([self.actor_params_push_op, self.critic_params_push_op], feed_dict=feed_dict)

    def sync_with_global(self):
        self.sess.run([self.actor_params_pull_op, self.critic_params_pull_op])

    def choose_action(self, s, cell_state):
        s = np.asarray(s).reshape(-1, self.n_features)
        cell_state = np.asarray(cell_state).reshape(1, self.cell_size)
        return self.sess.run([self.A, self.final_state], feed_dict={self.s: s, self.init_state: cell_state})


class Worker(object):
    def __init__(self, scope, globalAC: ACNet):
        self.scope = scope
        self.env: gym.Env = gym.make(GAME).unwrapped
        self.globalAC = globalAC
        self.AC = ACNet(scope=scope,
                        is_global=False,
                        globalAC=globalAC)
        self._reset_buffer()

    def _reset_buffer(self):
        self.buffer_s = []
        self.buffer_a = []
        self.buffer_r = []

    def work(self, COORD):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()  # 作为序列起始点，每次更新global时使用
            ep_r = 0
            rnn_state = self.AC.sess.run(self.AC.init_state)
            keep_state = rnn_state.copy()
            for ep_t in range(MAX_EP_STEP):
                a, rnn_state_ = self.AC.choose_action(s=s, cell_state=rnn_state)
                s_, r, done, info = self.env.step(a)  # need a (1,) shape
                done = True if ep_t == MAX_EP_STEP - 1 else False  # 最后一个循环为done
                ep_r += r
                self.buffer_s.append(s)
                self.buffer_a.append(a)
                self.buffer_r.append((r + 8) / 8)  # normalized to (-1, 1)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    v_s_ = 0 if done \
                        else np.asscalar(self.AC.sess.run(self.AC.v, feed_dict={self.AC.s: [s_],
                                                                                self.AC.init_state: rnn_state_}))
                    # trans buffer_r to buffer_v_target by v(s) = gamma * v(s_) + r circularly
                    for i in range(len(self.buffer_r) - 1, -1, -1):
                        v_s_ = v_s_ * self.AC.gamma + self.buffer_r[i]
                        self.buffer_r[i] = v_s_
                    # maybe misleading buffer_r is buffer_v_target now
                    self.AC.update_global(s=self.buffer_s, a=self.buffer_a,
                                          v_target=self.buffer_r, cell_state=keep_state)  # 更新
                    self.AC.sync_with_global()
                    self._reset_buffer()
                    keep_state = rnn_state_.copy()
                s = s_
                total_step += 1
                rnn_state = rnn_state_

                if done:
                    if not GLOBAL_RUNNING_R:
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print("%s  E.P.: %d  EP_r: %i" % (self.scope, GLOBAL_EP, GLOBAL_RUNNING_R[-1]))
                    GLOBAL_EP += 1
                    break


if __name__ == "__main__":
    tf.reset_default_graph()
    graph = tf.Graph()
    with tf.device("/CPU:0"):  # 都放在内存里
        with graph.as_default():
            GLOBAL_AC = ACNet(scope="GLOBAL", is_global=True)
            workers = []
            for i in range(multiprocessing.cpu_count()):
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
        worker.AC.sess = SESS
        t = threading.Thread(target=lambda: worker.work(COORD=COORD))
        t.start()
        worker_threads.append(t)
    COORD.join(threads=worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel("step")
    plt.ylabel("Total moving reward")
    plt.show()










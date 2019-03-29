"""
Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

Based on https://morvanzhou.github.io/tutorials

Using CLIP Penalty
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import queue
import threading


GAME = 'Pendulum-v0'
EP_MAX = 1000
EP_LEN = 200
N_WORKERS = 4
GAMMA = 0.9
LR_A = 0.0001
LR_C = 0.0002
BATCH_SIZE = 64
UPDATE_STEPS = 10
EPSILON = 0.2

env: gym.Env = gym.make(GAME).unwrapped
N_A = env.action_space.shape[0]
N_S = env.observation_space.shape[0]
A_BOUND = [env.action_space.low[0], env.action_space.high[0]]


class PPO(object):
    # sess = None
    n_actions = N_A
    n_features = N_S
    critic_lr = LR_C
    actor_lr = LR_A
    action_low, action_high = A_BOUND
    update_steps = UPDATE_STEPS
    epsilon = EPSILON

    def __init__(self, output_graph: bool=True):
        self.graph = self._build_graph()
        self.sess = self._build_session(self.graph)
        self.sess.run(self.init)
        if output_graph:
            tf.summary.FileWriter("F:/board/DPPO", self.graph)

    def _build_session(self, graph):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
        config.gpu_options.allow_growth = True  # 程序按需申请内存
        sess = tf.Session(graph=graph, config=config)
        return sess

    def _build_policy_net(self, scope: str, trainable: bool):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(inputs=self.s, units=100, activation=tf.nn.relu,
                                 kernel_initializer=self.w_init,
                                 bias_initializer=self.b_init,
                                 trainable=trainable,
                                 name="l1")

            mu = tf.layers.dense(inputs=l1, units=self.n_actions, activation=tf.nn.tanh,
                                 kernel_initializer=self.w_init,
                                 bias_initializer=self.b_init,
                                 trainable=trainable,
                                 name="mu") * self.action_high

            sigma = tf.layers.dense(inputs=l1, units=self.n_actions, activation=tf.nn.softplus,
                                    kernel_initializer=self.w_init,
                                    bias_initializer=self.b_init,
                                    trainable=trainable,
                                    name="sigma")

            normal_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="PolicyNet/" + scope)
        return normal_dist, params

    def _build_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            self.w_init = tf.random_normal_initializer(mean=0.0, stddev=0.1)  # if stddev=0.3 will be worse, wtf!
            self.b_init = tf.constant_initializer(value=0.0)

            with tf.variable_scope("input"):
                self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="State")
                self.a = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name="Action")
                self.v_ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="V_target")

            with tf.variable_scope("ValueNet"):  # critic
                l1 = tf.layers.dense(inputs=self.s, units=100, activation=tf.nn.relu,
                                     kernel_initializer=self.w_init,
                                     bias_initializer=self.b_init,
                                     name="l1")

                self.v = tf.layers.dense(inputs=l1, units=1, activation=None,
                                         kernel_initializer=self.w_init,
                                         bias_initializer=self.b_init,
                                         name="V")
                td_error = self.v_ - self.v  # advantage
                with tf.name_scope("loss"):
                    value_loss = tf.reduce_mean(tf.square(td_error))
                with tf.name_scope("train"):
                    self.critic_train_op = tf.train.AdamOptimizer(self.critic_lr).minimize(value_loss)

            with tf.variable_scope("PolicyNet"):  # actor

                pi, pi_params = self._build_policy_net(scope="Pi", trainable=True)
                old_pi, old_pi_params = self._build_policy_net(scope="Old_Pi", trainable=False)

                with tf.name_scope("Choose_Action"):
                    self.A = tf.clip_by_value(tf.squeeze(pi.sample(sample_shape=1), axis=[0, 1]),
                                              clip_value_min=self.action_low,
                                              clip_value_max=self.action_high)

                with tf.name_scope("Update_Old_Pi"):
                    self.update_old_pi_op = [old_p.assign(p) for old_p, p in zip(old_pi_params, pi_params)]

                with tf.name_scope("loss"):
                    with tf.name_scope("surrogate"):
                        ratio = pi.prob(self.a) / old_pi.prob(self.a)
                        adv = tf.stop_gradient(td_error, name="Advantage")
                        surr = ratio * adv
                        """
                        Clip Penalty objective => max L = E[min(ratio * Adv, clip(ratio, 1-eps, 1+eps) * Adv)]
                        """
                    policy_loss = -tf.reduce_mean(
                        tf.minimum(
                            surr,   # element-wise compare
                            tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
                        )
                    )
                with tf.name_scope("train"):
                    self.actor_train_op = tf.train.AdamOptimizer(self.actor_lr).minimize(policy_loss)

            self.init = tf.global_variables_initializer()
        return graph

    def learn(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()   # waiting for the data collection
                self.sess.run(self.update_old_pi_op)
                data = np.vstack([QUEUE.get() for _ in range(QUEUE.qsize())])
                s = data[:, :self.n_features]
                a = data[:, self.n_features: self.n_features + self.n_actions]
                v_ = data[:, -1:]
                for _ in range(self.update_steps):
                    self.sess.run(self.actor_train_op, feed_dict={self.s: s, self.a: a, self.v_: v_})
                for _ in range(self.update_steps):
                    self.sess.run(self.critic_train_op, feed_dict={self.s: s, self.v_: v_})

                UPDATE_EVENT.clear()
                GLOBAL_UPDATE_COUNTER = 0
                ROLLING_EVENT.set()

    def choose_action(self, s):
        s = np.asarray(s).reshape(-1, self.n_features)
        return self.sess.run(self.A, feed_dict={self.s: s})


class Worker(object):
    def __init__(self, wid: int):
        self.id = wid
        self.env: gym.Env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO
        self._reset_buffer()

    def _reset_buffer(self):
        self.buffer_s = []
        self.buffer_a = []
        self.buffer_r = []

    def _pack_data(self):
        return np.hstack(map(np.vstack, (self.buffer_s, self.buffer_a, self.buffer_r)))

    def work(self):
        global GLOBAL_EP, GLOBAL_UPDATE_COUNTER, GLOBAL_RUNNING_R
        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            for _ in range(EP_LEN):
                if not ROLLING_EVENT.is_set():       # while global PPO is updating
                    ROLLING_EVENT.wait()             # wait until PPO is updated
                    self._reset_buffer()             # clear old buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                self.buffer_s.append(s)
                self.buffer_a.append(a)
                self.buffer_r.append((r + 8) / 8)
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1           # collect batch alone, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= BATCH_SIZE:
                    v_s_ = self.ppo.sess.run(self.ppo.v, feed_dict={self.ppo.s: s_[np.newaxis, :]})
                    for i in range(len(self.buffer_r) - 1, -1, -1):
                        v_s_ = v_s_ * GAMMA + self.buffer_r[i]
                        self.buffer_r[i] = v_s_

                    # put data into global queue
                    data = self._pack_data()
                    QUEUE.put(data)
                    self._reset_buffer()

                    if GLOBAL_UPDATE_COUNTER >= BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break
            if not GLOBAL_RUNNING_R:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
            GLOBAL_EP += 1
            print("Progress: {0:.1f}%".format(GLOBAL_EP / EP_MAX * 100), " |Worker: %d  |Ep_r: %.2f" % (self.id, ep_r))


if __name__ == '__main__':
    GLOBAL_PPO = PPO(output_graph=True)
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()
    UPDATE_EVENT = threading.Event()   # ppo's learning event
    ROLLING_EVENT = threading.Event()  # worker's collect event
    UPDATE_EVENT.clear()               # init Set False
    ROLLING_EVENT.set()                # init Set True

    GLOBAL_UPDATE_COUNTER = 0
    GLOBAL_EP = 0
    GLOBAL_RUNNING_R = []

    workers = [Worker(wid=i) for i in range(N_WORKERS)]
    threads = []

    # Add task for threads
    for worker in workers:
        t = threading.Thread(target=worker.work, args=())
        t.start()
        threads.append(t)

    learn_task = threading.Thread(target=GLOBAL_PPO.learn)
    learn_task.start()
    threads.append(learn_task)

    COORD.join(threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel("Episode")
    plt.ylabel("Moving Averaged Reward")
    plt.show()

    while True:
        s = env.reset()
        for t in range(300):
            env.render()
            s = env.step(GLOBAL_PPO.choose_action(s))[0]

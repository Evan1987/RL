"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

Based on https://morvanzhou.github.io/tutorials
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym


GAME = 'Pendulum-v0'
EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
LR_A = 0.0001
LR_C = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10

env: gym.Env = gym.make(GAME).unwrapped
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low[0], env.action_space.high[0]]

METHOD = [
    {"name": "KL", "kl_target": 0.01, "beta": 0.5},  # KL Penalty
    {"name": "CLIP", "epsilon": 0.2}  # Clipped Surrogate Objective, better one
][1]


class PPO(object):
    # sess = None
    n_actions = N_A
    n_features = N_S
    critic_lr = LR_C
    actor_lr = LR_A
    action_low, action_high = A_BOUND
    actor_update_steps = A_UPDATE_STEPS
    critic_update_steps = C_UPDATE_STEPS

    def __init__(self, method: dict, output_graph: bool=True):
        self.method = method
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
                    if self.method["name"] == "KL":
                        """
                        Adaptive KL Penalty  objective => max L = E[ratio * Adv - beta * KL]
                        """
                        kl = tf.distributions.kl_divergence(old_pi, pi)
                        self.kl_mean = tf.reduce_mean(kl)  # to adjust beta value
                        policy_loss = -tf.reduce_mean(surr - self.method["beta"] * kl)
                    else:
                        """
                        Clip Penalty objective => max L = E[min(ratio * Adv, clip(ratio, 1-eps, 1+eps) * Adv)]
                        """
                        policy_loss = -tf.reduce_mean(
                            tf.minimum(
                                surr,   # element-wise compare
                                tf.clip_by_value(ratio, 1.0 - self.method["epsilon"], 1.0 + self.method["epsilon"]) * adv
                            )
                        )
                with tf.name_scope("train"):
                    self.actor_train_op = tf.train.AdamOptimizer(self.actor_lr).minimize(policy_loss)

            self.init = tf.global_variables_initializer()
        return graph

    def learn(self, s, a, v_target):
        self.sess.run(self.update_old_pi_op)  # Synchronise old_pi to pi
        feed_dict = {self.s: s, self.a: a, self.v_: v_target}

        # Update Actor
        if self.method["name"] == "KL":
            for _ in range(self.actor_update_steps):
                _, kl = self.sess.run([self.actor_train_op, self.kl_mean], feed_dict=feed_dict)
                if kl > 4 * self.method["kl_target"]:  # by google paper
                    break
            # adaptive in OpenAI's paper
            if kl < self.method["kl_target"] / 1.5:
                self.method["beta"] /= 2
            elif kl > self.method["kl_target"] * 1.5:
                self.method["beta"] *= 2
            self.method["beta"] = np.clip(self.method["beta"], 1e-4, 10)
        else:
            for _ in range(self.actor_update_steps):
                self.sess.run(self.actor_train_op, feed_dict=feed_dict)

        # Update Critic

        for _ in range(self.critic_update_steps):
            self.sess.run(self.critic_train_op, feed_dict={self.s: s, self.v_: v_target})

    def choose_action(self, s):
        s = np.asarray(s).reshape(-1, self.n_features)
        return self.sess.run(self.A, feed_dict={self.s: s})


if __name__ == "__main__":

    env: gym.Env = gym.make(GAME).unwrapped
    ppo = PPO(method=METHOD)
    all_ep_r = []

    for ep in range(EP_MAX):
        s = env.reset()
        buffer_r, buffer_s, buffer_a = [], [], []
        ep_r = 0
        for t in range(EP_LEN):
            #env.render()
            a = ppo.choose_action(s)
            s_, r, done, _ = env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r + 8) / 8)  # normalized to [-1, 1]
            s = s_
            ep_r += r

            if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
                v_s_ = np.asscalar(ppo.sess.run(ppo.v, feed_dict={ppo.s: s_[np.newaxis, :]}))
                for i in range(len(buffer_r) - 1, -1, -1):
                    v_s_ = v_s_ * GAMMA + buffer_r[i]
                    buffer_r[i] = v_s_

                ppo.learn(s=np.vstack(buffer_s), a=np.vstack(buffer_a), v_target=np.asarray(buffer_r).reshape(-1, 1))

                buffer_r, buffer_s, buffer_a = [], [], []

        if not all_ep_r:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        print("E.P. %d  EP_r: %i" % (ep, all_ep_r[-1]),
              "   beta: %.4f" % METHOD["beta"] if METHOD["name"] == "KL" else "")

    plt.plot(range(len(all_ep_r)), all_ep_r)
    plt.xlabel("Episode")
    plt.ylabel("Moving Averaged Episode Reward")
    plt.show()











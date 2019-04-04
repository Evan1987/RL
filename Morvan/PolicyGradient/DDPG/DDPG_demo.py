"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
Based on https://morvanzhou.github.io/tutorials
"""

import tensorflow as tf
import numpy as np
import gym


np.random.seed(1)
tf.set_random_seed(1)

MAX_EP = 200
MAX_EP_STEPS = 200
LR_A = 0.001
LR_C = 0.001
GAMMA = 0.9
REPLACEMENT = [
    {"name": "soft", "tau": 0.01},
    {"name": "hard", "rep_iter": 500}
][0]
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
GAME = 'Pendulum-v0'
OUTPUT_GRAPH = True

env: gym.Env = gym.make(GAME).unwrapped
N_A = env.action_space.shape[0]
N_S = env.observation_space.shape[0]
ACTION_LOW = env.action_space.low
ACTION_HIGH = env.action_space.high


class DDPG(object):
    actor_lr = LR_A
    critic_lr = LR_C
    memory_size = MEMORY_CAPACITY
    n_actions = N_A
    n_features = N_S
    action_low = ACTION_LOW
    action_high = ACTION_HIGH
    replacement_method = REPLACEMENT
    gamma = GAMMA
    batch_size = BATCH_SIZE

    def __init__(self):
        self.memory = np.zeros(shape=(self.memory_size, 2 * self.n_features + self.n_actions + 1), dtype=np.float32)
        self.pointer = 0
        self.graph = self._build_graph()
        self.sess = self._build_session(graph=self.graph)
        self.sess.run(self.init)
        if OUTPUT_GRAPH:
            tf.summary.FileWriter("F:/board/DDPG", graph=self.graph)

    def _build_session(self, graph):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
        config.gpu_options.allow_growth = True  # 程序按需申请内存
        sess = tf.Session(graph=graph, config=config)
        return sess

    def _build_actor_net(self, s, scope: str, trainable: bool):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(inputs=s, units=30, activation=tf.nn.relu, name="l1", trainable=trainable)
            a = tf.layers.dense(inputs=l1, units=self.n_actions, activation=tf.nn.tanh, trainable=trainable)
            return tf.multiply(a, self.action_high, name="scaled_action")

    def _build_critic_net(self, s, a, scope: str, trainable: bool):
        """
        Simulate Q(s, a) rather than V(s)
        """
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable(name="w1_s", shape=[self.n_features, n_l1], trainable=trainable)
            w1_a = tf.get_variable(name="w1_a", shape=[self.n_actions, n_l1], trainable=trainable)
            b1 = tf.get_variable(name="b1", shape=[1, n_l1], trainable=trainable)
            l1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(inputs=l1, units=1, activation=None, trainable=trainable)

    def _build_graph(self):
        graph = tf.Graph()
        tf.reset_default_graph()
        with graph.as_default():
            with tf.name_scope("input"):
                self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="s")
                self.s_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="s_")
                self.r = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="reward")

            with tf.variable_scope("Actor"):
                self.a = self._build_actor_net(self.s, scope="eval", trainable=True)
                a_ = self._build_actor_net(self.s_, scope="target", trainable=False)

            with tf.variable_scope("Critic"):
                q = self._build_critic_net(self.s, self.a, scope="eval", trainable=True)
                q_ = self._build_critic_net(self.s_, a_, scope="target", trainable=False)

            with tf.name_scope("Params_Replacement"):
                self.actor_eval_params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/eval")
                self.actor_target_params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/target")
                self.critic_eval_params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="Critic/eval")
                self.critic_target_params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="Critic/target")

                eval_params = self.actor_eval_params + self.critic_eval_params
                target_params = self.actor_target_params + self.critic_target_params
                if self.replacement_method["name"] == "hard":
                    self.param_replacement = [tf.assign(t, e)
                                              for t, e in zip(target_params, eval_params)]
                else:
                    tau = self.replacement_method["tau"]
                    self.param_replacement = [tf.assign(t, (1 - tau) * t + tau * e)
                                              for t, e in zip(target_params, eval_params)]

            with tf.name_scope("loss"):
                q_target = q_ * self.gamma + self.r
                td_error = tf.reduce_mean(tf.squared_difference(q_target, q))
                self.critic_train = tf.train.AdamOptimizer(self.critic_lr)\
                    .minimize(td_error, var_list=self.critic_eval_params)

                actor_loss = -tf.reduce_mean(q)  # to max Q
                self.actor_train = tf.train.AdamOptimizer(self.actor_lr)\
                    .minimize(actor_loss, var_list=self.actor_eval_params)

            self.init = tf.global_variables_initializer()

        return graph

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.a, feed_dict={self.s: s})[0]

    def store_transition(self, s, a, r, s_):
        transition = np.r_[s, a, [r], s_]  # hstack
        index = self.pointer % self.memory_size
        try:
            self.memory[index, :] = transition
        except Exception as e:
            print(transition)
            print(transition.shape)
            raise e
        self.pointer += 1

    def learn(self):
        self.sess.run(self.param_replacement)

        indices = np.random.choice(self.memory_size, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.n_features]
        ba = bt[:, self.n_features: self.n_features + self.n_actions]
        br = bt[:, self.n_features + self.n_actions].reshape(-1, 1)
        bs_ = bt[:, -self.n_features:]

        self.sess.run(self.actor_train, feed_dict={self.s: bs})
        self.sess.run(self.critic_train, feed_dict={self.s: bs, self.a: ba, self.s_: bs_, self.r: br})


if __name__ == "__main__":
    ddpg = DDPG()
    var = 3  # control exploration
    for i in range(MAX_EP):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(loc=a, scale=var), ACTION_LOW[0], ACTION_HIGH[0])
            s_, r, done, _ = env.step(a)

            ddpg.store_transition(s, a, (r + 8) / 8, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= 0.9995  # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r

            if j == MAX_EP_STEPS - 1:
                print("E.P.: %d, Reward: %.4f,  Explore: %.2f" % (i, ep_reward, var))

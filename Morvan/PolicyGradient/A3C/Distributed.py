"""
Based on https://morvanzhou.github.io/tutorials/

Another Edition of  Discrete A3C on Distributed Machines

@comment: Unchecked
"""

import gym
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

GAME = "CartPole-v0"
UPDATE_GLOBAL_ITER = 10
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
MAX_GLOBAL_EP = 1000
N_WORKERS = mp.cpu_count()
OUTPUT_GRAPH = True

#
# def _build_session(graph):
#     config = tf.ConfigProto()
#     config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
#     config.gpu_options.allow_growth = True  # 程序按需申请内存
#     sess = tf.Session(graph=graph, config=config)
#     return sess


class ACNet(object):
    sess = None

    def __init__(self, *, scope, n_actions, n_features, reward_decay=0.9, entropy_beta=0.001, actor_learning_rate=0.001,
                 critic_learning_rate=0.01, is_global: bool=False, globalAC=None):
        self.scope = scope
        self.n_actions = n_actions
        self.n_features = n_features
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
            acts_prob_logit = tf.layers.dense(inputs=a_l1, units=self.n_actions, activation=None,
                                              kernel_initializer=self.w_init,
                                              bias_initializer=self.b_init,
                                              name="acts_prob")

        actor_params = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope=self.scope + "/actor")
        critic_params = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope=self.scope + "/critic")
        return v, acts_prob_logit, actor_params, critic_params

    def _build_graph(self):
        self.w_init = tf.random_normal_initializer(mean=0, stddev=0.1)
        self.b_init = tf.constant_initializer(0.0)
        if self.is_global:
            with tf.variable_scope(self.scope):
                self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="S")
                _, _, self.actor_params, self.critic_params = self._build_net()
        else:
            with tf.variable_scope(self.scope):
                self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="S")
                self.a_his = tf.placeholder(dtype=tf.int32, shape=[None, ], name="A")
                self.v_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="V_target")
                self.v, acts_prob_logit, self.actor_params, self.critic_params = self._build_net()
                # dim=-1 is also ok which indicates last dim
                self.acts_prob = tf.nn.softmax(acts_prob_logit, dim=1)

                td = tf.subtract(self.v_target, self.v, name="td_error")  # tf.subtract element-wise x - y
                with tf.name_scope("critic_loss"):
                    # mse tf.reduce_mean(tf.squared_difference(x, y))
                    critic_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope("actor_loss"):
                    neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=acts_prob_logit,
                                                                                  labels=self.a_his)
                    weighted_neg_log_prob = neg_log_prob * tf.stop_gradient(td)
                    entropy = tf.reduce_sum(self.acts_prob * tf.log(self.acts_prob + 1e-5), axis=1, keep_dims=True)
                    actor_loss = tf.reduce_mean(-entropy * self.entropy_beta + weighted_neg_log_prob)

                with tf.name_scope("local_grad"):
                    self.actor_grads = tf.gradients(actor_loss, self.actor_params)
                    self.critic_grads = tf.gradients(critic_loss, self.critic_params)

                with tf.name_scope("opt"):
                    actor_opt = tf.train.RMSPropOptimizer(learning_rate=self.actor_lr)
                    critic_opt = tf.train.RMSPropOptimizer(learning_rate=self.critic_lr)

            self.global_step = tf.train.get_or_create_global_step()
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
                        actor_opt.apply_gradients(zip(self.actor_grads, self.globalAC.actor_params),
                                                  global_step=self.global_step)
                    self.critic_params_push_op = \
                        critic_opt.apply_gradients(zip(self.critic_grads, self.globalAC.critic_params))

    def update_global(self, *, s, a, v_target):
        s = np.asarray(s).reshape(-1, self.n_features)
        a = np.asarray(a, dtype=np.int32).ravel()
        v_target = np.asarray(v_target).reshape(-1, 1)
        feed_dict = {self.s: s, self.a_his: a, self.v_target: v_target}
        self.sess.run([self.actor_params_push_op, self.critic_params_push_op], feed_dict=feed_dict)

    def sync_with_global(self):
        self.sess.run([self.actor_params_pull_op, self.critic_params_pull_op])

    def choose_action(self, s):
        s = np.asarray(s)[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, feed_dict={self.s: s})
        return np.random.choice(range(self.n_actions), p=probs.ravel())


def work(job_name: str, task_index: int, global_ep: mp.Queue,
         lock: mp.Lock, r_queue: mp.Queue, global_running_r: mp.Queue):

    assert job_name in ["ps", "worker"], "Invalid job_name: %s" % job_name
    cluster = tf.train.ClusterSpec({
        "ps": ["localhost:2220", "localhost:2221"],
        "worker": ["localhost:2222", "localhost:2223", "localhost:2224", "localhost:2225"]
    })
    server = tf.train.Server(server_or_cluster_def=cluster, job_name=job_name, task_index=task_index)

    if job_name == "ps":
        print("Start Parameter Server: ", task_index)
        server.join()
    else:
        tic = time.time()
        env: gym.Env = gym.make(GAME).unwrapped
        n_actions = env.action_space.n,
        n_features = env.observation_space.shape[0]
        print("Start Worker: ", task_index)
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):
            # opt_a = tf.train.RMSPropOptimizer(LR_A, name='opt_a')
            # opt_c = tf.train.RMSPropOptimizer(LR_C, name='opt_c')
            global_ac = ACNet(scope="global_net",
                              n_actions=n_actions,
                              n_features=n_features,
                              is_global=True)
            local_net = ACNet(scope="local_ac%d" % task_index,
                              n_actions=n_actions,
                              n_features=n_features,
                              globalAC=global_ac)
        hooks = [tf.train.StopAtStepHook(last_step=100000)]

        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=True, hooks=hooks) as sess:
            print("Start Worker Session: ", task_index)
            local_net.sess = sess
            total_step = 1
            buffer_s, buffer_a, buffer_r = [], [], []
            while (not sess.should_stop()) and (global_ep.value < 1000):
                s = env.reset()
                ep_r = 0
                done = False
                while not done:
                    a = local_net.choose_action(s)
                    s_, r, done, _ = env.step(a)
                    if done:
                        r = -5
                    ep_r += r
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append(r)

                    if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                        v_s_ = 0 if done else np.asscalar(sess.run(local_net.v, feed_dict={local_net.s: [s_]}))
                        buffer_v_target = []
                        for r in buffer_r[::-1]:  # reverse buffer r
                            v_s_ = r + local_net.gamma * v_s_
                            buffer_v_target.append(v_s_)
                        buffer_v_target.reverse()
                        local_net.update_global(s=buffer_s, a=buffer_a, v_target=buffer_v_target)
                        local_net.sync_with_global()
                        buffer_s, buffer_a, buffer_r = [], [], []
                    s = s_
                    total_step += 1

                    if done:
                        if r_queue.empty():
                            global_running_r.value = ep_r
                        else:
                            global_running_r.value = 0.99 * global_running_r.value + 0.01 * ep_r
                        r_queue.put(global_running_r.value)
                        global_step = sess.run(local_net.global_step)
                        print("Task %d | E.P. %d | Ep_r %i | Global_Step %d "
                              % (task_index, global_ep.value, global_running_r.value, global_step))

                        with lock:
                            global_ep.value += 1
        toc = time.time()
        print("Worker %d Done! Use time %i" % (task_index, toc - tic))


if __name__ == "__main__":
    # using multiprocessing to create a local cluster with 2 parameter servers and 4 workers
    global_ep = mp.Value("i", 0)
    lock = mp.Lock
    r_queue = mp.Queue()
    global_running_r = mp.Value("d", 0)

    jobs = [("ps", 0), ("ps", 1),
            ("worker", 0), ("worker", 1), ("worker", 2), ("worker", 3)]

    for name, index in jobs:
        p = mp.Process(target=work, kwargs={"job_name": name, "task_index": index, "global_ep": global_ep,
                                            "lock": lock, "r_queue": r_queue, "global_running_r": global_running_r})
        p.start()
        if name == "worker":
            p.join()

    ep_r = []
    while not r_queue.empty():
        ep_r.append(r_queue.get())

    plt.plot(np.arange(len(ep_r)), ep_r)
    plt.title('Distributed training')
    plt.xlabel('Step')
    plt.ylabel('Total moving reward')
    plt.show()



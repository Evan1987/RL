"""
This part of code followed book "the Essence of RL"
"""

import numpy as np
import random
import time
from Snake import Snakes
from TableAgent import TableAgent, eval_game
from contextlib import contextmanager


@contextmanager
def timer(name, verbose: int=1):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        if verbose:
            print("%s COST: %.6f" % (name, end - start))


class PolicyIteration(object):
    def __init__(self, agent: TableAgent, max_iter=-1, epsilon=1e-6):
        """
        策略迭代算法
        :param agent: 定义的 Agent类型
        :param max_iter: Vπ的最大迭代数
        :param epsilon: 精度阈值，判断是否收敛
        """
        self.agent = agent
        self.max_iter = max_iter
        self.epsilon = epsilon

    def rmse(self, x: np.ndarray, y: np.ndarray) -> float:
        return sum(np.power(x - y, 2)) ** 0.5

    def policy_evaluation(self):
        """
        更新 agent的状态值函数 value_pi, Vπ(S)
        切记：动作是 选择哪个骰子，而不是掷出的步数
        表格式的策略表示注定了动作选择概率 π是非0即1
        """
        iteration = 0
        while iteration != self.max_iter:
            iteration += 1
            new_value_pi = self.agent.value_pi.copy()
            for i in range(1, self.agent.s_len):  # 遍历全部status
                # 当前策略（旧的，更新值函数时保持不变），由于是表格式记载了所选择的动作，所以动作的选择概率是非0即1的。
                # 即 π(一（二）号骰子|S) = 0 或 1
                action = self.agent.pi[i]
                transition = self.agent.p[action, i, :]
                new_value_pi[i] = np.asscalar(np.dot(transition, self.agent.r + self.agent.gamma * self.agent.value_pi))

            if self.rmse(new_value_pi, self.agent.value_pi) < self.epsilon:  # 如果收敛则停止
                break
            self.agent.value_pi = new_value_pi
        return iteration

    def policy_improvement(self) -> bool:
        """
        更新 agent的行为值函数 value_q(qπ)，以及策略 π
        :return: 策略是否发生了变化
        """
        # 遍历更新q的每一个元素
        for i in range(1, self.agent.s_len):
            for act in range(self.agent.a_len):
                transition = self.agent.p[act, i, :]
                self.agent.value_q[i, act] = np.asscalar(np.dot(transition, self.agent.r + self.agent.gamma * self.agent.value_pi))
        new_pi = np.argmax(self.agent.value_q, axis=1)  # 策略更新

        if np.all(np.equal(new_pi, self.agent.pi)):  # 策略是否发生了变化
            return False

        self.agent.pi = new_pi
        return True

    def policy_iteration(self):
        """策略迭代主函数"""
        epoch = 0
        ret = True
        while ret:
            epoch += 1
            num_iters = self.policy_evaluation()
            ret = self.policy_improvement()
            print("Epoch %d: loops %d rounds" % (epoch, num_iters))
        print("Iter %d rounds converge!" % epoch)


class ValueIteration(object):
    def __init__(self, agent: TableAgent, max_iter=-1, epsilon=1e-6):
        """
        策略迭代算法
        :param agent: 定义的 Agent类型
        :param max_iter: Vπ的最大迭代数
        :param epsilon: 精度阈值，判断是否收敛
        """
        self.agent = agent
        self.max_iter = max_iter
        self.epsilon = epsilon

    def rmse(self, x: np.ndarray, y: np.ndarray) -> float:
        return sum(np.power(x - y, 2)) ** 0.5

    def value_iteration(self):
        iteration = 0
        while iteration != self.max_iter:
            iteration += 1
            new_value_pi = np.zeros_like(self.agent.value_pi)
            for i in range(1, self.agent.s_len):
                max_value = -float("inf")
                for act in range(self.agent.a_len):
                    transition = self.agent.p[act, i, :]
                    value = np.asscalar(np.dot(transition, self.agent.r + self.agent.gamma * self.agent.value_pi))
                    max_value = max(max_value, value)
                new_value_pi[i] = max_value
            if self.rmse(new_value_pi, self.agent.value_pi) < self.epsilon:
                break
            self.agent.value_pi = new_value_pi
        print("Iter %d rounds converge!" % iteration)

        for i in range(1, self.agent.s_len):
            for act in range(self.agent.a_len):
                transition = self.agent.p[act, i, :]
                self.agent.value_q[i, act] = np.asscalar(np.dot(transition, self.agent.r + self.agent.gamma * self.agent.value_pi))
            self.agent.pi = np.argmax(self.agent.value_q, axis=1)


class GeneralizedPolicyIteration(object):
    def __init__(self, agent: TableAgent, max_value_iter=10, max_policy_iter=1):
        self.agent = agent
        self.pi_alg = PolicyIteration(self.agent, max_iter=max_policy_iter)
        self.vi_alg = ValueIteration(self.agent, max_iter=max_value_iter)

    def generalized_policy_iteration(self):
        self.vi_alg.value_iteration()
        self.pi_alg.policy_iteration()


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    env = Snakes(ladder_num=10, dices=[3, 6])
    eval_num = 10000
    print(env.ladders)

    policies = {
        "All_0": [0] * 100,
        "All_1": [1] * 100,
        "Randomly": np.random.randint(low=0, high=2, size=100).tolist(),
        "Opt": [1] * 97 + [0] * 3,
        "Agent": TableAgent(env)
    }

    final_pi = None

    for name, policy in policies.items():
        sum_ = 0
        if isinstance(policy, list):
            for _ in range(eval_num):
                sum_ += eval_game(env, policy)
            score = sum_ / eval_num
            print("%s policy avg. score: %.2f" % (name, score))
        elif isinstance(policy, TableAgent):
            print("*" * 10)
            with timer("PolicyIteration", 1):
                pi_alg = PolicyIteration(policy, max_iter=-1)
                pi_alg.policy_iteration()
                final_pi = pi_alg.agent.pi
            score = eval_game(env, policy)
            print("%s policy avg. score: %.2f" % ("PolicyIteration", score))
            print(final_pi)

            print("*" * 10)
            with timer("ValueIteration", 1):
                vi_alg = ValueIteration(policy, max_iter=-1)
                vi_alg.value_iteration()
                final_pi = vi_alg.agent.pi
            score = eval_game(env, policy)
            print("%s policy avg. score: %.2f" % ("ValueIteration", score))
            print(final_pi)

            print("*" * 10)
            with timer("GeneralizedPolicyIteration", 1):
                g_pi_alg = GeneralizedPolicyIteration(policy, max_policy_iter=1, max_value_iter=10)
                g_pi_alg.generalized_policy_iteration()
                final_pi = g_pi_alg.agent.pi
            score = eval_game(env, policy)
            print("%s policy avg. score: %.2f" % ("GeneralizedPolicyIteration", score))
            print(final_pi)



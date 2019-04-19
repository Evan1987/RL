"""
This part of code followed book "the Essence of RL"
"""

import abc
import numpy as np
from Agent import TableAgent


class Alg(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def iteration(self):
        pass


class PolicyIteration(Alg):
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
        self.iter_counter = 0

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
            new_value_pi = np.zeros_like(self.agent.value_pi)
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

    def iteration(self):
        """策略迭代主函数"""
        ret = True
        while ret:
            self.iter_counter += self.policy_evaluation()
            ret = self.policy_improvement()


class ValueIteration(Alg):
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
        self.iter_counter = 0

    def rmse(self, x: np.ndarray, y: np.ndarray) -> float:
        return sum(np.power(x - y, 2)) ** 0.5

    def iteration(self):
        while self.iter_counter != self.max_iter:
            self.iter_counter += 1
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

        for i in range(1, self.agent.s_len):
            for act in range(self.agent.a_len):
                transition = self.agent.p[act, i, :]
                self.agent.value_q[i, act] = np.asscalar(np.dot(transition, self.agent.r + self.agent.gamma * self.agent.value_pi))
        self.agent.pi = np.argmax(self.agent.value_q, axis=1)


class GeneralizedPolicyIteration(Alg):
    def __init__(self, agent: TableAgent, max_value_iter=10, max_policy_iter=1):
        self.agent = agent
        self.pi_alg = PolicyIteration(self.agent, max_iter=max_policy_iter)
        self.vi_alg = ValueIteration(self.agent, max_iter=max_value_iter)
        self.iter_counter = 0

    def iteration(self):
        self.vi_alg.iteration()
        self.pi_alg.iteration()
        self.iter_counter += self.vi_alg.iter_counter + self.pi_alg.iter_counter







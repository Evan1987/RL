"""
This part of code followed book "the Essence of RL"
"""

import numpy as np
import random
from Snake import Snakes
from TableAgent import TableAgent, eval_game


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

    def policy_evaluation(self) -> None:
        """
        更新 agent的状态值函数 value_pi, Vπ(S)
        切记：动作是 选择哪个骰子，而不是掷出的步数
        表格式的策略表示注定了动作选择概率 π是非0即1
        """
        iteration = 0
        while iteration != self.max_iter:
            iteration += 1
            value_pi = self.agent.value_pi.copy()
            for i in range(1, self.agent.s_len):  # 遍历全部status
                # 当前策略（旧的，更新值函数时保持不变），由于是表格式记载了所选择的动作，所以动作的选择概率是非0即1的。
                # 即 π(一（二）号骰子|S) = 0 或 1
                action = self.agent.pi[i]
                transition = self.agent.p[action, i, :]
                value_pi[i] = np.asscalar(np.dot(transition, self.agent.r + self.agent.gamma * self.agent.value_pi))

            if self.rmse(value_pi, self.agent.value_pi) < self.epsilon:  # 如果收敛则停止
                break
            self.agent.value_pi = value_pi

    def policy_improvement(self) -> bool:
        """
        更新 agent的行为值函数 value_q(qπ)，以及策略 π
        :return: 策略是否发生了变化
        """
        # 遍历更新q的每一个元素
        for i in range(1, self.agent.s_len):
            for j in range(self.agent.a_len):
                transition = self.agent.p[j, i, :]
                self.agent.value_q[i, j] = np.asscalar(np.dot(transition, self.agent.r + self.agent.gamma * self.agent.value_pi))
        new_pi = np.argmax(self.agent.value_q, axis=1)  # 策略更新

        if np.all(np.equal(new_pi, self.agent.pi)):  # 策略是否发生了变化
            return False

        self.agent.pi = new_pi
        return True

    def policy_iteration(self):
        """策略迭代主函数"""
        iteration = 0
        ret = True
        while ret:
            iteration += 1
            self.policy_evaluation()
            ret = self.policy_improvement()
        print("Iter %d rounds converge!" % iteration)


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    env = Snakes(ladder_num=5, dices=[3, 6])
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
        elif isinstance(policy, TableAgent):
            alg = PolicyIteration(policy)
            alg.policy_iteration()
            final_pi = alg.agent.pi
            score = eval_game(env, policy)

        print("%s policy avg. score: %.2f" % (name, score))
        if final_pi is not None:
            print(final_pi)



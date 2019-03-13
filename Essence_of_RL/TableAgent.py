"""
This part of code followed book "the Essence of RL"
"""

import numpy as np


class TableAgent(object):
    def __init__(self, env, *, gamma=0.8):
        self.s_len = env.observation_space.n  # 101
        self.a_len = env.action_space.n  # 2
        self.dices = env.dices  # length: 2
        self.ladders = env.ladders

        self.r = np.array([env.reward(s) for s in range(self.s_len)])
        self.pi = np.zeros(self.s_len, dtype=np.int32)  # 策略（列表式地呈现） length: 101，每个状态选择哪个骰子
        self.value_pi = np.zeros(self.s_len, dtype=np.float32)  # 状态值函数 length: 101, 在某策略下每个状态的预期价值
        self.value_q = np.zeros((self.s_len, self.a_len))  # 行为值函数 101 * 2
        self.p = self.init_p()  # 存储状态转移情况 P(S`|S, A)  2 * 101 * 101  A * |St| * |St+1|

        self.gamma = gamma

    def move(self, pos):
        pos = pos if pos <= 100 else 200 - pos
        return pos if pos not in self.ladders else self.ladders[pos]

    def init_p(self) -> np.ndarray:
        p = np.zeros(shape=[self.a_len, self.s_len, self.s_len], dtype=np.float32)
        for i, dice in enumerate(self.dices):  # 选择骰子
            prob = 1 / dice
            steps = np.arange(1, dice + 1)
            for src in range(1, self.s_len - 1):  # 遍历每一个状态 [1 -> 99]
                dst_list = [self.move(pos) for pos in src + steps]
                for dst in dst_list:
                    p[i, src, dst] += prob
        p[:, 100, 100] = 1.0
        return p

    def play(self, state):
        return self.pi[state]


def eval_game(env, policy):
    """
    在env下测试策略效果
    :param env:  game环境
    :param policy: 策略。 TableAgent类型 或 列表(length 与 state数量相同)
    :return: 此种策略下的总得分
    """
    assert isinstance(policy, list) or isinstance(policy, TableAgent), "Illegal Policy"
    policy = policy.pi if isinstance(policy, TableAgent) else policy

    state = env.reset()
    total_reward = 0
    while True:
        act = policy[state]
        state, reward, done, _ = env.step(act)
        total_reward += reward
        if done:
            break
    return total_reward






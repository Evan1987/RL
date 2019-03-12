"""
This part of code followed book "the Essence of RL"
"""

import numpy as np
import Snake

class TableAgent(object):
    def __init__(self, env):
        self.s_len = env.observation_space.n  # 101
        self.a_len = env.action_space.n  # 2
        self.dices = env.dices  # length: 2
        self.ladders = env.ladders

        self.r = [env.reward(s) for s in range(self.s_len)]
        self.pi = self.value_pi = [0] * len(self.s_len)  # 策略列表  length: 101
        self.p = self.init_p()  # 存储状态转移情况 P(S`|S, A)  2 * 101 * 101
        self.q = np.zeros((self.s_len, self.a_len))  # Q表 101 * 2

    def ladder_move(self, pos):
        return pos if pos not in self.ladders else self.ladders[pos]

    def init_p(self):
        p = np.zeros(shape=[self.a_len, self.s_len, self.s_len], dtype=np.float32)
        for i, dice in enumerate(self.dices):  # 选择骰子
            prob = 1 / dice
            for src in range(1, self.s_len - 1):  # 遍历每一个状态 [1 -> 99]
                steps = np.arange(1, dice + 1)
                dsts = src + steps
                dsts = [self.ladder_move(pos) for pos in dsts]
                for dst in dsts:
                    p[i, src, dst] += prob
        p[:, 100, 100] = 1.0
        return p

    def play(self, state):
        return self.pi[state]


def eval_game(env, policy):
    """

    :param env:  game环境
    :param policy: 策略。 TableAgent类型 或 列表(length 与 state数量相同)
    :return: 此种策略下的总得分
    """
    assert isinstance(policy, list) or isinstance(policy, TableAgent), "Illegal Policy"
    choose_act = lambda state: policy.play(state) if isinstance(policy, TableAgent) else lambda state: policy[state]

    state = env.reset()
    total_reward = 0
    while True:
        act = choose_act(state)
        state, reward, done, _ = env.step(act)
        total_reward += reward
        if done:
            break
    return total_reward


if __name__ == '__main__':
    env = Snake.Snakes(ladder_num=0, dices=[3, 6])

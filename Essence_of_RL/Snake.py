
"""
This part of code followed book "the Essence of RL"
"""

import numpy as np
import random
import gym
import time
from gym.spaces import Discrete


class Snakes(gym.Env):
    SIZE = 100

    def __init__(self, *, ladders=None, ladder_num=5, dices=[3, 6]):  # * 后面的参数必须加关键字
        """
        :param ladders: 自定义指定的ladder连接信息，字典形式双向化
        :param ladder_num: 不指定ladders时，需要随机产生的ladder数量
        :param dices:  可选择的骰子，用骰子的最大值代表
        """
        super().__init__()
        self.dices = dices
        self.ladders = ladders if ladders and isinstance(ladders, dict) else self.set_ladders(ladder_num)
        self.observation_space = Discrete(self.SIZE + 1)
        self.action_space = Discrete(len(dices))
        self.pos = 1

    def set_ladders(self, ladder_num):
        ladders = {}
        points = np.random.choice(range(1, self.SIZE), size=2 * ladder_num, replace=False).reshape(-1, 2)
        for x, y in points:
            ladders[x] = y
            ladders[y] = x
        return ladders

    def reset(self):
        self.pos = 1
        return 1

    def step(self, act):
        d = self.dices[act]
        choice = random.choice(range(1, d + 1))
        self.pos += choice
        if self.pos == 100:
            return 100, 100.0, True, {}
        elif self.pos > 100:
            self.pos = 200 - self.pos

        if self.pos in self.ladders:
            self.pos = self.ladders[self.pos]

        return self.pos, -1.0, False, {}

    def render(self, mode='human'):
        pass

    @staticmethod
    def reward(s):
        if s == 100:
            return 100.0
        return -1.0


if __name__ == "__main__":
    dices = [3, 6]
    env = Snakes(ladder_num=10, dices=dices)
    env.reset()

    score = i = 0
    while True:
        env.render()
        action = random.choice(range(len(dices)))  # 随机选择一个骰子
        state, reward, is_done, _ = env.step(action)
        score += reward
        i += 1
        print("Step: %d  Pos: %d  Score: %d" % (i, state, score))
        time.sleep(0.2)
        if is_done:
            print("*" * 20)
            print("Total Step: %d  Score: %d" % (i, score))
            break

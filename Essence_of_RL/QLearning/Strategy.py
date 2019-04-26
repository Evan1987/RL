
"""
This part of code followed book "the Essence of RL"
"""

import numpy as np
import os
from typing import Optional
from Policy import Alg
from Snake import Snakes
from Agent import ModelFreeAgent


class MonteCarlo(Alg):
    def __init__(self, env: Snakes, agent: ModelFreeAgent, *, epsilon=0.8,
                 eval_iter=100, max_iter=10, episode_save: Optional[str]=None):
        """
        使用 monte-carlo方法完成策略迭代
        :param env: 游戏环境
        :param agent: agent
        :param epsilon: agent的 e-greedy参数
        :param eval_iter: 内层策略评估的迭代次数
        :param max_iter: 外层策略迭代次数
        :param episode_save: 采样信息保存地址
        """
        self.env = env
        self.agent = agent
        self.agent.e_greedy = epsilon
        self.eval_iter = eval_iter
        self.max_iter = max_iter
        self.episode_save = episode_save

    def policy_eval(self):
        """策略评估"""
        for _ in range(self.eval_iter):
            state = self.env.reset()
            episode = []  # 记录状态行为序列，元素为三元组[si, ai, r]
            done = False
            while not done:  # monte-carlo采样，不断交互
                act = self.agent.play(state)
                next_state, reward, done, _ = self.env.step(act)
                episode.append((state, act, reward))
                state = next_state

            # 将每个状态的当期回报，变成预期回报（未来回报 * 各级筛减因子 + 当期回报）
            return_val = 0.0
            for i in range(len(episode) - 1, -1, -1):
                return_val = self.agent.gamma * return_val + episode[i][2]
                episode[i] = (episode[i][0], episode[i][1], return_val)

            if self.episode_save is not None:
                save_path = os.path.join(self.episode_save)
                with open(save_path, "a+") as f:
                    for state, act, return_val in episode:
                        f.write("%d\t%d\t%.4f\n" % (state, act, return_val))

            # 根据大数定理，利用均值（期望）给q(s,a)赋值
            for state, act, return_val in episode:
                self.agent.value_n[state, act] += 1
                # 均值更新
                self.agent.value_q[state, act] += \
                    (return_val - self.agent.value_q[state, act]) / self.agent.value_n[state, act]

    def policy_improve(self):
        """策略更新"""
        new_pi = np.argmax(self.agent.value_q, axis=1)
        if np.all(np.equal(new_pi, self.agent.pi)):
            return False
        self.agent.pi = new_pi
        return True

    def iteration(self):
        """整体迭代"""
        for _ in range(self.max_iter):
            self.policy_eval()
            self.policy_improve()


class SARSA(Alg):
    def __init__(self, env: Snakes, agent: ModelFreeAgent, *, epsilon=0.8, max_iter=10, eval_iter=200):
        """
        使用时序差分进行策略迭代，参数说明与 monte-carlo一致
        """
        self.env = env
        self.agent = agent
        self.agent.e_greedy = epsilon
        self.max_iter = max_iter
        self.eval_iter = eval_iter

    def policy_eval(self):
        """策略评估"""
        for _ in range(self.eval_iter):
            state = self.env.reset()
            prev_state = -1  # s
            prev_act = -1  # a
            done = False
            while not done:
                act = self.agent.play(state)  # state: s`, act: a`
                next_state, reward, done, _ = self.env.step(act)  # reward: r(s`), next_state: new s`
                if prev_act >= 0:
                    # r(s`) + gamma * q(s`,a`)
                    return_val = reward + self.agent.gamma * (0 if done else self.agent.value_q[state, act])
                    self.agent.value_n[prev_state, prev_act] += 1

                    # q(s, a) += {[r(s`) + gamma * q(s`, a`)] - q(s, a)} / N
                    self.agent.value_q[prev_state, prev_act] += \
                        (return_val - self.agent.value_q[prev_state, prev_act]) / self.agent.value_n[prev_state, prev_act]

                prev_act = act
                prev_state = state
                state = next_state

    def policy_improve(self):
        """策略更新"""
        new_pi = np.argmax(self.agent.value_q, axis=1)
        if np.all(np.equal(new_pi, self.agent.pi)):
            return False
        self.agent.pi = new_pi
        return True

    def iteration(self):
        """整体迭代"""
        for _ in range(self.max_iter):
            self.policy_eval()
            self.policy_improve()


class QLearning(Alg):
    def __init__(self, env: Snakes, agent: ModelFreeAgent, *, epsilon=0.8, max_iter=10, eval_iter=200):
        self.env = env
        self.agent = agent
        self.agent.e_greedy = epsilon
        self.max_iter = max_iter
        self.eval_iter = eval_iter

    def policy_eval(self):
        for _ in range(self.eval_iter):
            state = self.env.reset()
            prev_state = -1  # s
            prev_act = -1  # a
            done = False
            while not done:
                act = self.agent.play(state)  # state: s`, act: a`
                next_state, reward, done, _ = self.env.step(act)  # reward: r(s`), next_state: new s`
                if prev_act >= 0:
                    # r(s`) + gamma * max(q(s`,a`))
                    return_val = reward + self.agent.gamma * (0 if done else self.agent.value_q[state, :].max())
                    self.agent.value_n[prev_state, prev_act] += 1

                    # q(s, a) += {[r(s`) + gamma * q(s`, a`)] - q(s, a)} / N
                    self.agent.value_q[prev_state, prev_act] += \
                        (return_val - self.agent.value_q[prev_state, prev_act]) / self.agent.value_n[prev_state, prev_act]

                prev_act = act
                prev_state = state
                state = next_state

    def policy_improve(self):
        new_pi = np.argmax(self.agent.value_q, axis=1)
        if np.all(np.equal(new_pi, self.agent.pi)):
            return False
        self.agent.pi = new_pi
        return True

    def iteration(self):
        for _ in range(self.max_iter):
            self.policy_eval()
            self.policy_improve()

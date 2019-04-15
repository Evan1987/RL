
"""
This part of code followed book "the Essence of RL"
"""

import numpy as np
import os
from Policy import Alg
from Snake import Snakes
from Agent import ModelFreeAgent


class MonteCarlo(Alg):
    def __init__(self, env: Snakes, agent: ModelFreeAgent, *, epsilon=0.8, eval_iter=100, max_iter=10, episode_save=None):
        self.env = env
        self.agent = agent
        self.agent.e_greedy = epsilon
        self.eval_iter = eval_iter
        self.max_iter = max_iter
        self.episode_save = episode_save

    def policy_eval(self):
        for _ in range(self.eval_iter):
            state = self.env.reset()
            episode = []  # 记录状态行为序列，元素为三元组[si, ai, r]
            done = False
            while not done:
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

            for state, act, return_val in episode:
                self.agent.value_n[state, act] += 1
                # 均值更新
                self.agent.value_q[state, act] += \
                    (return_val - self.agent.value_q[state, act]) / self.agent.value_n[state, act]

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


class SARSA(Alg):
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
        new_pi = np.argmax(self.agent.value_q, axis=1)
        if np.all(np.equal(new_pi, self.agent.pi)):
            return False
        self.agent.pi = new_pi
        return True

    def iteration(self):
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

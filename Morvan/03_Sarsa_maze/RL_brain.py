"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
"""

import numpy as np
import pandas as pd


class RL:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=actions, dtype=np.float64)

    def check_state_exist(self, state):
        """
        if state not in q_table, append q_table
        """
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(data=[0] * len(self.actions), index=self.actions, name=state)
            )

    def choose_action(self, state):
        self.check_state_exist(state)

        rand = np.random.uniform()
        if rand < self.epsilon:
            state_action = self.q_table.loc[state, :]
            # the max value may be multiple. Randomly choose
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


class QLearningTable(RL):
    """
    Off-policy
    """
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super().__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)


class SarsaTable(RL):
    """
    On-policy
    """
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super().__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)

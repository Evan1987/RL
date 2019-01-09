"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6  # length of the 1-dimension-world
ACTIONS = ["left", "right"]  # available actions
EPSILON = 0.9  # greedy policy
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODES = 13  # maximum episodes
FRESH_TIME = 0.3  # fresh time for one move

def build_q_table(n_states, actions):
    """
    :param n_states: length of world
    :param actions:  available choices
    :return: choice table, shape: [n_states, #actions]
    """
    return pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)

def choose_action(state, q_table):
    """
    How to choose an action at some state based on q_table
    :param state: now state
    :param q_table:  q_table
    :return: action chosen at the state
    """
    state_actions = q_table.iloc[state, :]  # series with length #actions
    rand = np.random.uniform()  # a random number bewteen [0, 1]
    if ((state_actions == 0).all()) or (rand > EPSILON):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()  # choose the action with biggest value
    return action_name

def get_env_feedback(S, A):
    """
    Agent interact with Env.
    The treasure is placed at right wall
    :param S: state
    :param A: action at state S
    :return: next state, value for thie state action
    """

    if A == "right":
        if S == N_STATES - 2:  # bingo!
            S_ = "terminal"
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # move left
        R = 0
        S_ = max(S - 1, 0)
    return S_, R

def update_env(S, episode, step_count):
    """
    Env update
    :param S:
    :param episode:
    :param step_counter:
    :return:
    """
    env_list = ["-"] * (N_STATES - 1) + ["T"]  # "------T" stands for our env
    if S == "terminal":
        interaction = "Episode %d: total_steps: %d\n" % (episode + 1, step_count)
        print(interaction)
        time.sleep(2)
        print("**********************************************")
    else:
        env_list[S] = "o"
        interaction = "".join(env_list)
        print(interaction)
        time.sleep(FRESH_TIME)

if __name__ == "__main__":

    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_count = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_count)

        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # get next state and feedback for this action A
            q_predict = q_table.loc[S, A]  # pred value for S, A without feedback
            if S_ != "terminal":
                q_target = R + GAMMA * q_table.iloc[S_, :].max()  # evaluated value for S_
            else:
                q_target = R  # the next state is terminal
                is_terminated = True

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update q_table -> a memory
            S = S_

            step_count += 1
            update_env(S, episode, step_count)

    print(q_table)

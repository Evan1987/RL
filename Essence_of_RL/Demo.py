
"""
This part of code followed book "the Essence of RL"
"""

import numpy as np
import random
from Snake import Snakes
from Policy import PolicyIteration, ValueIteration, GeneralizedPolicyIteration
from QLearning.MonteCarlo import MonteCarlo
from Agent import TableAgent, eval_game, ModelFreeAgent
from Utils import timer


def alg_test(alg_name, alg, n_iters: int):
    print("*" * 10)
    sum_ = 0
    with timer(alg_name, 1):
        for _ in range(n_iters):
            alg.iteration()
            sum_ += eval_game(env, alg.agent)
    print("%s policy avg. score: %.2f" % (alg_name, sum_ / n_iters))
    print(alg.agent.pi)


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
        "Opt": [1] * 97 + [0] * 3
    }

    for name, policy in policies.items():
        sum_ = 0
        for _ in range(eval_num):
            sum_ += eval_game(env, policy)
        print("%s policy avg. score: %.2f" % (name, sum_ / eval_num))

    print("*" * 20)
    algs = {
        "PolicyIteration": PolicyIteration(TableAgent(env), max_iter=-1),
        "ValueIteration": ValueIteration(TableAgent(env), max_iter=-1),
        "GeneralizedPolicyIteration": GeneralizedPolicyIteration(TableAgent(env), max_policy_iter=1, max_value_iter=10),
        "MonteCarlo": MonteCarlo(env, ModelFreeAgent(env), eval_iter=100, max_iter=10)
    }

    for alg_name, alg in algs.items():
        alg_test(alg_name, alg, 100)

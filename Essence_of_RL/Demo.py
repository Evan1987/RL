
"""
This part of code followed book "the Essence of RL"
"""

import numpy as np
import pandas as pd
import random
from Snake import Snakes
from Policy import PolicyIteration, ValueIteration, GeneralizedPolicyIteration
from QLearning.Strategy import MonteCarlo, SARSA, QLearning
from Agent import TableAgent, eval_game, ModelFreeAgent
from Utils import timer
from _utils.u_constant import PATH_ROOT

save_path = PATH_ROOT + "Code projects/Python/RL_Learn/Essence_of_RL/QLearning/"
monte_carlo_samples_path = save_path + "monte_carlo_samples.txt"
monte_carlo_samples_std_path = save_path + "monte_carlo_samples_std.txt"


def alg_test(alg_name, alg, n_iters: int):
    print("*" * 20)
    sum_ = 0
    with timer(alg_name, 1):
        for _ in range(n_iters):
            alg.iteration()
            sum_ += eval_game(env, alg.agent)
    print("%s policy avg. score: %.2f" % (alg_name, sum_ / n_iters))
    print(alg.agent.pi)


def monte_carlo_std(sample_path):
    df = pd.read_table(sample_path, names=["status", "act", "val"])
    res = df.groupby(["status", "act"], as_index=False).agg({"val": ["count", "std"]})
    res.columns = [x + "_" + y if y else x for x, y in res.columns]
    return res


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
        "MonteCarlo": MonteCarlo(env, ModelFreeAgent(env), eval_iter=100,  max_iter=10, episode_save=monte_carlo_samples_path),
        "SARSA": SARSA(env, ModelFreeAgent(env), max_iter=10, eval_iter=100, epsilon=0.5),
        "QLearning": QLearning(env, ModelFreeAgent(env), max_iter=10, eval_iter=100, epsilon=0.5)
    }

    for alg_name, alg in algs.items():
        alg_test(alg_name, alg, 100)

    res = monte_carlo_std(monte_carlo_samples_path)
    res.to_csv(monte_carlo_samples_std_path, index=False, sep="\t")



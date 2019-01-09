"""
Sarsa is a on-policy updating method for Reinforcement learning.

Unlike Q learning which is a off-policy updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""


import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import time
from maze_env import Maze
from RL_brain import SarsaTable

if __name__ == '__main__':
    env = Maze()
    RL = SarsaTable(actions=env.action_space)
    def update():
        log = []
        for episode in range(100):
            s = env.reset()
            a = RL.choose_action(str(s))
            step_count = 0
            done = False
            r = 0
            while not done:
                env.render()
                s_, r, done = env.step(a)
                a_ = RL.choose_action(str(s_))

                RL.learn(str(s), a, r, str(s_), a_)
                s = s_
                a = a_
                step_count += 1
                time.sleep(0.02)

            result = "Win" if r == 1 else "Failed"
            log.append((episode + 1, step_count, result))
            print("Episode: %d  Total_steps: %d %s!" % (episode + 1, step_count, result))

        print("Game Over!")
        df = pd.DataFrame(log, columns=["episode", "total_step", "result"])
        df.to_excel(os.getcwd() + "/sarsa_log.xlsx", index=False)
        env.destroy()

    env.after(100, update)
    env.mainloop()

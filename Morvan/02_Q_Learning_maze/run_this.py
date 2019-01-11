"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
"""


import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import time
from maze_env import Maze
from RL_brain import QLearningTable

if __name__ == '__main__':
    path = os.getcwd()
    env = Maze()
    RL = QLearningTable(actions=env.action_space)
    
    def update():
        log = []
        for episode in range(30):
            s = env.reset()
            step_count = 0
            done = False
            r = 0
            while not done:
                env.render()
                a = RL.choose_action(str(s))
                s_, r, done = env.step(a)

                RL.learn(str(s), a, r, str(s_))
                s = s_
                step_count += 1
                time.sleep(0.02)

            result = "Win" if r == 1 else "Failed"
            log.append((episode + 1, step_count, result))
            print("Episode: %d  Total_steps: %d %s!" % (episode + 1, step_count, result))

        print("Game Over!")
        df = pd.DataFrame(log, columns=["episode", "total_step", "result"])
        df.to_excel(path + "/qlearn_log.xlsx", index=False)
        env.destroy()

    env.after(100, update)
    env.mainloop()
    RL.q_table.to_excel(path + "/q_table.xlsx", index=True)

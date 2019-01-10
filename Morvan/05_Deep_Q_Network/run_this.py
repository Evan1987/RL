
import sys
import os
sys.path.append(os.getcwd())
from maze_env import Maze
from RL_brain import DeepQNetwork


if __name__ == '__main__':
    env = Maze()
    RL = DeepQNetwork(env.n_actions,
                      env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=200)

    def run_maze():
        step = 0
        for episode in range(300):
            s = env.reset()
            done = False

            while not done:
                env.render()

                a = RL.choose_action(s)
                s_, r, done = env.step(a)
                
                # DQN 存储记忆
                RL.store_transition(s, a, r, s_)

                # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
                if (step > 200) and (step % 5 == 0):
                    RL.learn()

                s = s_
                step += 1
        print("Game Over")
        env.destroy()


    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()

import sys
import os
sys.path.append(os.getcwd())
import gym
from Essence_of_RL.QLearning.DQN import DeepQNetWork


RENDER = False
MAX_EP_STEPS = 1000
DISPLAY_REWARD_THRESHOLD = 200

if __name__ == '__main__':
    env = gym.make("CartPole-v0").unwrapped
    RL = DeepQNetWork(env.action_space.n,
                      env.observation_space.shape[0],
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.8,
                      replace_target_iter=200,
                      double_q=True,  # Double DQN
                      prioritized=False,  # use PRB
                      dueling=True,  # use Dueling DQN
                      memory_size=200,
                      output_graph=True)

    running_reward = None
    epoch = 0
    for i_episode in range(3000):
        s = env.reset()
        t = 0
        track_r = []
        while True:
            if RENDER:
                env.render()
            a = RL.choose_action(s)
            s_, r, done, info = env.step(a)

            if done:
                r = -20
            track_r.append(r)
            RL.store_transition(s, a, r, s_)

            s = s_
            t += 1
            epoch += 1

            # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
            if (epoch > 200) and (epoch % 5 == 0):
                RL.learn()

            if done or t >= MAX_EP_STEPS:
                # calculate running reward
                ep_rs_sum = sum(track_r)
                if not running_reward:
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True
                print("Episode: %d  Reward: %d" % (i_episode, int(running_reward)))
                break
    RL.plot_cost()

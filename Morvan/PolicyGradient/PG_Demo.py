"""
Based on https://morvanzhou.github.io/tutorials/
"""

import gym
from Morvan.PolicyGradient.RL_brain import PolicyGradient
import matplotlib.pyplot as plt

RENDER = False

DISPLAY_REWARD_THRESHOLD_MAP = {"CartPole-v0": 300, "MountainCar-v0": -2000}

# renders environment if total episode reward is greater then this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502

game_id = "CartPole-v0"
env = gym.make(game_id)  # when change
DISPLAY_REWARD_THRESHOLD = DISPLAY_REWARD_THRESHOLD_MAP[game_id]

env.seed(1)
env: gym.Env = env.unwrapped

RL = PolicyGradient(n_actions=env.action_space.n,
                    n_features=env.observation_space.shape[0],
                    learning_rate=0.02,
                    reward_decay=0.995)


if __name__ == '__main__':

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    running_reward = None

    for i_episode in range(1000):
        s = env.reset()
        n_round = 0
        while True:
            n_round += 1
            if RENDER:
                env.render()
            a = RL.choose_action(s)
            s_, r, done, info = env.step(a)
            RL.store_transition(s, a, r)

            if done:
                # calculate running reward
                ep_rs_sum = sum(RL.rb_r)
                if not running_reward:
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True
                print("Episode: %d  Reward: %d" % (i_episode, int(running_reward)))

                vt = RL.learn()

                if i_episode == 30:
                    plt.plot(vt)
                    plt.xlabel("episode steps")
                    plt.ylabel("normalized state-action value")
                    plt.show()

                break

            s = s_

"""
Based on https://morvanzhou.github.io/tutorials/

with td_error as advantage
"""


import gym
from Morvan.PolicyGradient.RL_brain import ActorCritic

RENDER = False
MAX_EP_STEPS = 1000
DISPLAY_REWARD_THRESHOLD_MAP = {"CartPole-v0": 200, "MountainCar-v0": -2000}

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

RL = ActorCritic(n_actions=env.action_space.n,
                 n_features=env.observation_space.shape[0],
                 actor_learning_rate=0.001,
                 critic_learning_rate=0.01,
                 reward_decay=0.9,
                 output_graph=True)


if __name__ == '__main__':

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    running_reward = None

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
            RL.learn(s, a, r, s_)

            s = s_
            t += 1

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



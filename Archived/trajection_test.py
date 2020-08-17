import gym
import gym_fightingice
import random

env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path="..", port=6666, p2="ReiwaThunder", frameskip=False, use_sim=False)
env.reset()
action_list = [i for i in range(56)]
while True:
    s_prime, r, done, info = env.step(random.choice(action_list))
    valid_actions = info.get('my_action_enough', {})
    action_list = []
    for i in range(56):
        if i in valid_actions:
            action_list.append(i)

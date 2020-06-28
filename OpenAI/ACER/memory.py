# -*- coding: utf-8 -*-
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'policy', 'action_mask'))


class EpisodicReplayMemory():
    # TODO: add the TD error as the priority
    def __init__(self, capacity, max_episode_length):
        # Max number of transitions possible will be the memory capacity, could be much less
        self.num_episodes = int(capacity)
        self.memory = deque(maxlen=self.num_episodes)
        self.trajectory = []

    def append_transition(self, state, action, reward, policy, action_mask, discard=False):
        self.trajectory.append(Transition(state, action, reward, policy, action_mask))  # Save s_i, a_i, r_i+1, µ(·|s_i)
        # Terminal states are saved with actions as None, so switch to next episode
        if action is None:
            if not discard:
                self.memory.append(self.trajectory)
            self.trajectory = []

    def append_trajectory(self, trajectory):
        self.memory.append(trajectory)

    # Samples random trajectory
    def sample(self, maxlen=0):
        mem = self.memory[random.randrange(len(self.memory))]
        T = len(mem)
        # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
        if maxlen > 0 and T > maxlen + 1:
            t = random.randrange(T - maxlen - 1)  # Include next state after final "maxlen" state
            return mem[t:t + maxlen + 1]
        else:
            return mem

    def last_trajectory(self):
        return self.memory[-1]  # return the last trajectory

    # Samples batch of trajectories, truncating them to the same length
    def sample_batch(self, batch_size, maxlen=0):
        batch = [self.sample(maxlen=maxlen) for _ in range(batch_size)]
        minimum_size = min(len(trajectory) for trajectory in batch)
        batch = [trajectory[:minimum_size] for trajectory in batch]  # Truncate trajectories
        return list(map(list, zip(*batch)))  # Transpose so that timesteps are packed together

    def on_policy(self):
        batch = [self.memory[-1]]
        return list(map(list, zip(*batch)))  # Transpose so that timesteps are packed together

    def length(self):
        # Return number of epsiodes saved in memory
        return len(self.memory)

    def __len__(self):
        return len(self.memory)

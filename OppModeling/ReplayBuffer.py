import torch
import numpy as np
from OppModeling.utils import combined_shape

fields = ('obs', 'next_obs', 'action', 'reward', 'done', 'latent')


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store_trajectory(self, trajectory):
        for i in trajectory:
            self.store(i["obs"], i["action"], i["reward"], i["next_obs"], i["done"])

    def sample_batch(self, batch_size=32,device=None):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}


class ReplayBufferShare:
    """
    A simple FIFO experience replay buffer for shared memory.
    """

    def __init__(self, buffer, size):
        self.buffer = buffer
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(dict(obs=obs, next_obs=next_obs, action=act, reward=rew, done=done))
        else:
            self.buffer.pop(0)
            self.buffer.append(dict(obs=obs, next_obs=next_obs, action=act, reward=rew, done=done))
        self.ptr = (self.ptr + 1) % self.max_size

    def sample_batch(self, batch_size=32, device=None):
        idxs = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idxs]
        obs_buf, obs2_buf, act_buf, rew_buf, done_buf = [], [], [], [], []
        for trans in batch:
            obs_buf.append(trans["obs"])
            obs2_buf.append(trans["next_obs"])
            act_buf.append(trans["action"])
            rew_buf.append(trans["reward"])
            done_buf.append(trans["done"])
        batch_dict = dict(obs=obs_buf, obs2=obs2_buf, act=act_buf, rew=rew_buf, done=done_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch_dict.items()}


class ReplayBufferOppo:
    # for single thread or created in the child thread
    def __init__(self, max_size, encoder):
        self.trajectories = list()
        self.traj_len = list()
        self.encoder = encoder
        self.z_dim = self.encoder.z_dim
        self.c_dim = self.encoder.c_dim
        self.max_size = max_size

    def store(self, trajectory):
        self.trajectories.append(trajectory)
        self.traj_len.append(len(trajectory))
        self.forget()

    def forget(self):
        pass

    def cluster(self):
        pass

    def update_latent(self, trajectories, use_gpu=True):
        batch_size = len(trajectories)
        for trans in trajectories:
            z_hidden = self.encoder.init_hidden(batch_size, self.z_dim, use_gpu=True)
            c_hidden = self.encoder.init_hidden(batch_size, self.c_dim, use_gpu=True)
        output, encoder_hidden, c_hidden = self.encoder.pre

    def sample_trans(self, batch_size, device=None):
        indexes = np.arange(len(self.trajectories))
        prob = self.traj_len
        sampled_traj_index = np.random.choice(indexes, size=batch_size, replace=True, p=prob)
        sampled_trans = [np.random.choice(self.trajectories[index]) for index in sampled_traj_index]
        obs_buf, obs2_buf, act_buf, rew_buf, done_buf = [],[],[],[],[]
        for trans in sampled_trans:
            obs_buf.append(trans["obs"] + trans["latent"])
            obs2_buf.append(trans["next_obs"])
            act_buf.append(trans["action"])
            rew_buf.append(trans["reward"])
            done_buf.append(trans["done"])
        batch = dict(obs=obs_buf, obs2=obs2_buf, act=act_buf, rew=rew_buf,done=done_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}

    def sample_traj(self,batch_size, max_seq_len):
        indexes = np.random.randint(len(self.trajectories), size=batch_size)
        min_len = [self.traj_len[i] for i in indexes]
        # cut off using the min length
        batch = [self.trajectories[i][:min_len] for i in indexes]
        return batch



import gym
import argparse
import gym_fightingice
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.distributions import Categorical
from copy import deepcopy
from OpenAI.atari_wrappers import make_ftg_ram
from model_parameter_trans import state_dict_trans,load_trajectory

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class Counter():
    def __init__(self):
        self.val = mp.Value('i', 0)
        self.lock = mp.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def set(self, value):
        with self.lock:
            self.val.value = value

    def value(self):
        with self.lock:
            return self.val.value


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation=nn.Identity):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation)

    def forward(self, obs):
        net_out = self.net(obs)
        a_prob = F.softmax(net_out, dim=1).clamp(min=1e-20, max=1-1e-20)
        return a_prob


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation=nn.Identity):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation)

    def forward(self, obs):
        net_out = self.net(obs)
        return net_out


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n
        # act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation)
        self.q1 = Critic(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = Critic(obs_dim, act_dim, hidden_sizes, activation)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self,data, ac_targ, gamma, alpha):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a_prob, log_a_prob, sample_a, max_a = self.get_actions_info(self.pi(o2))

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2)
            q2_pi_targ = ac_targ.q2(o2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (a_prob * (q_pi_targ - alpha * log_a_prob)).sum(dim=1)

        # MSE loss against Bellman backup
        q1 = self.q1(o).gather(1, a.unsqueeze(-1).long())
        q2 = self.q2(o).gather(1, a.unsqueeze(-1).long())
        loss_q1 = F.mse_loss(q1, backup.unsqueeze(-1))
        loss_q2 = F.mse_loss(q2, backup.unsqueeze(-1))
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data, alpha):
        o = data['obs']
        a_prob, log_a_prob, sample_a, max_a = self.get_actions_info(self.pi(o))
        q1_pi = self.q1(o)
        q2_pi = self.q2(o)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (a_prob * (alpha * log_a_prob - q_pi)).mean()
        entropy = torch.sum(log_a_prob * a_prob, dim=1).detach()

        # Useful info for logging
        pi_info = dict(LogPi=entropy.numpy())
        return loss_pi, pi_info

    def act(self, obs):
        with torch.no_grad():
            a_prob = self.pi(obs)
            return a_prob

    def get_action(self, o, greedy=False):
        if len(o.shape) == 1:
            o = np.expand_dims(o, axis=0)
        a_prob = self.act(torch.as_tensor(o, dtype=torch.float32))
        a_prob, log_a_prob, sample_a, max_a = self.get_actions_info(a_prob)
        action = sample_a if not greedy else max_a
        return action.item()

        # product action

    @staticmethod
    def get_actions_info(a_prob):
        a_dis = Categorical(a_prob)
        max_a = torch.argmax(a_prob)
        sample_a = a_dis.sample().cpu()
        z = a_prob == 0.0
        z = z.float() * 1e-8
        log_a_prob = torch.log(a_prob + z)
        return a_prob, log_a_prob, sample_a, max_a


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

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def sac(global_ac, global_ac_targ, rank, T, args,scores, ac_kwargs=dict(), env =None, p2 =None,seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, max_ep_len=1000,
        save_freq=1):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    env = make_ftg_ram(env, p2=p2)
    obs_dim = env.observation_space.shape[0]
    print("set up child process env")
    local_ac = MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    state_dict = global_ac.state_dict()
    local_ac.load_state_dict(state_dict)
    print("local ac load global ac")
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    # Async Version
    for p in global_ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=replay_size)
    # if args.traj_dir:
    #     replay_buffer.store_trajectory(load_trajectory(args.traj_dir))

    # Set up optimizers for policy and q-function
    # Async Version
    pi_optimizer = Adam(global_ac.pi.parameters(), lr=lr)
    q1_optimizer = Adam(global_ac.q1.parameters(), lr=lr)
    q2_optimizer = Adam(global_ac.q2.parameters(), lr=lr)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0
    discard = False
    t = T.value()
    # Main loop: collect experience in env and update/log each epoch
    while t <= total_steps:

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = local_ac.get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        if info.get('no_data_receive', False):
            discard = True
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if (ep_len == max_ep_len) or discard else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        T.increment()
        t = T.value()

        # End of trajectory handling
        if d or (ep_len == max_ep_len) or discard:
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            scores.append(ep_ret)
            m_score = np.mean(scores[-100:])
            print("Process {}, # of global_steps :{}, round score: {}, mean score : {:.1f}, steps: {}".format(
                rank, t, ep_ret, m_score, ep_len))
            o, ep_ret, ep_len = env.reset(), 0, 0
            discard = False

        # Update handling
        if t >= update_after and t % update_every == 0:
            print("update network")
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                # First run one gradient descent step for Q1 and Q2
                q1_optimizer.zero_grad()
                q2_optimizer.zero_grad()
                loss_q, q_info = local_ac.compute_loss_q(batch, global_ac_targ, gamma, alpha)
                loss_q.backward()

                # Next run one gradient descent step for pi.
                pi_optimizer.zero_grad()
                loss_pi, pi_info = local_ac.compute_loss_pi(batch, alpha)
                loss_pi.backward()

                for global_param, local_param in zip(global_ac.parameters(), local_ac.parameters()):
                    global_param._grad = local_param.grad

                pi_optimizer.step()
                q1_optimizer.step()
                q2_optimizer.step()

                state_dict = global_ac.state_dict()
                local_ac.load_state_dict(state_dict)

                # Finally, update target networks by polyak averaging.
                with torch.no_grad():
                    for p, p_targ in zip(global_ac.parameters(), global_ac_targ.parameters()):
                        p_targ.data.copy_((1 - polyak) * p.data + polyak * p_targ.data)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            print("Epoch: {}".format(epoch))
        if t % save_freq == 0 and t > 0:
            torch.save(global_ac.state_dict(), os.path.join(args.save_dir, "model"))
            state_dict_trans(global_ac.state_dict(), os.path.join(args.save_dir, args.numpy_para))
            print("Saving model at episode:{}".format(t))


if __name__ == '__main__':
    mp.set_start_method("forkserver")
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="FightingiceDataFrameskip-v0")
    parser.add_argument('--p2', type=str, default="ReiwaThunder")
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--n_process', type=int, default=4)
    parser.add_argument('--save-dir', type=str, default="./OpenAI/SAC")
    parser.add_argument('--traj_dir', type=str, default="./OpenAI/SAC")
    parser.add_argument('--numpy_para', type=str, default="sac.pkl")
    args = parser.parse_args()
    torch.set_num_threads(torch.get_num_threads())
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)
    env = make_ftg_ram(args.env, p2=args.p2)
    global_ac = MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    if os.path.exists(os.path.join(args.save_dir, "model")):
        global_ac.load_state_dict(torch.load(os.path.join(args.save_dir, "model")))
        # state_dict_trans(global_model.state_dict(), os.path.join(save_dir, numpy_para))
        print("load model")
    global_ac_targ = deepcopy(global_ac)
    env.close()
    del env
    global_ac.share_memory()
    global_ac_targ.share_memory()

    var_counts = tuple(count_vars(module) for module in [global_ac.pi, global_ac.q1, global_ac.q2])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    # this the kwargs for the single thread version


    T = Counter()
    scores = mp.Manager().list()
    processes = []
    p2_list = ["ReiwaThunder", "RHEA_PI", "Toothless", "FalzAI"]
    for rank in range(args.n_process):  # + 1 for test process
        # if rank == 0:
            # p = mp.Process(target=test, args=(global_model,))
        # else:
        p2 = p2_list[rank % len(p2_list)]
        single_version_kwargs = dict(ac_kwargs=ac_kwargs, env=args.env, p2=p2, gamma=args.gamma, seed=args.seed,
                                     epochs=args.epochs,
                                     steps_per_epoch=1000, replay_size=int(1e6),
                                     polyak=0.995, lr=args.lr, alpha=0.2, batch_size=128, start_steps=10000,
                                     update_after=10000, update_every=10, max_ep_len=500,
                                     save_freq=1000)
        p = mp.Process(target=sac, args=(global_ac, global_ac_targ, rank, T, args, scores), kwargs=single_version_kwargs)
        p.start()
        time.sleep(5)
        processes.append(p)
    for p in processes:
        p.join()




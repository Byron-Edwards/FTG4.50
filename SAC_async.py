import gym
import argparse
import gym_fightingice
import json
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
from collections import namedtuple
import random
# from tensorboardX import GlobalSummaryWriter
from torch.utils.tensorboard import SummaryWriter
from OpenAI.atari_wrappers import make_ftg_ram,make_ftg_ram_nonstation
from model_parameter_trans import state_dict_trans,load_trajectory

Transition = namedtuple('Transition', ('obs', 'next_obs', 'action', 'reward', 'done'))

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
    def __init__(self,obs_dim, act_dim, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        # build policy and value functions
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation)
        self.q1 = Critic(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = Critic(obs_dim, act_dim, hidden_sizes, activation)
        self.log_alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

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

        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data, alpha):
        o = data['obs']
        a_prob, log_a_prob, sample_a, max_a = self.get_actions_info(self.pi(o))
        q1_pi = self.q1(o)
        q2_pi = self.q2(o)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (a_prob * (alpha * log_a_prob - q_pi)).mean()
        entropy = torch.sum(log_a_prob * a_prob, dim=1)

        # Useful info for logging
        # pi_info = dict(LogPi=entropy.numpy())
        return loss_pi, entropy

    def act(self, obs):
        with torch.no_grad():
            a_prob = self.pi(obs)
            return a_prob

    def get_action(self, o, greedy=False, device=None):
        if len(o.shape) == 1:
            o = np.expand_dims(o, axis=0)
        a_prob = self.act(torch.as_tensor(o, dtype=torch.float32, device=device))
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

    def sample_batch(self, batch_size=32,device=None):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}


class ReplayBufferOppo:
    # for single thread or created in the child thread
    def __init__(self, max_size, oppo_mode= False):
        self.trajectories = list()
        self.traj_len = list()
        self.latent = list()
        self.oppo_mode = oppo_mode
        self.max_size = max_size

    def store(self, trajectory,encoder):
        self.trajectories.append(trajectory)
        self.traj_len.append(len(trajectory))
        self.latent.append(encoder(trajectory)[0])
        if not self.oppo_mode:
            # as a simple queue
            if len(self.trajectories) > self.max_size:
                self.trajectories.pop(0)
                self.traj_len.pop(0)
                self.latent.pop(0)
        else:
            self.forget()

    def forget(self):
        pass

    def cluster(self):
        pass

    def sample_trans(self,batch_size, device=None):
        indexes = np.arange(len(self.trajectories))
        prob = self.traj_len
        sampled_traj_index = np.random.choice(indexes, size=batch_size, replace=True, p=prob)
        sampled_trans = [np.random.choice(self.trajectories[index]) for index in sampled_traj_index]
        obs_buf,obs2_buf,act_buf,rew_buf,done_buf = [],[],[],[],[]
        for trans in sampled_trans:
            obs_buf.append(trans.obs)
            obs2_buf.append(trans.next_obs)
            act_buf.append(trans.action)
            rew_buf.append(trans.reward)
            done_buf.append(trans.done)
        batch = dict(obs=obs_buf,obs2=obs2_buf,act=act_buf,rew=rew_buf,done=done_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}

    def sample_traj(self,batch_size, max_seq_len):
        indexes = np.random.randint(len(self.trajectories), size=batch_size)
        min_len = [self.traj_len[i] for i in indexes]
        # cut off using the min length
        batch = [self.trajectories[i][:min_len] for i in indexes]
        return batch


class ReplayBufferShare:
    """
    A simple FIFO experience replay buffer for SAC async version.
    """

    def __init__(self, buffer, size):
        self.buffer = buffer
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(Transition(obs=obs, next_obs=next_obs, action=act, reward=rew, done=done))
        else:
            self.buffer.pop(0)
            self.buffer.append(Transition(obs=obs, next_obs=next_obs, action=act, reward=rew, done=done))
        self.ptr = (self.ptr + 1) % self.max_size

    def store_trajectory(self, trajectory):
        for i in trajectory:
            self.store(i["obs"], i["action"], i["reward"], i["next_obs"], i["done"])

    def sample_batch(self, batch_size=32, device=None):
        idxs = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idxs]
        obs_buf, obs2_buf, act_buf, rew_buf, done_buf = [], [], [], [], []
        for trans in batch:
            obs_buf.append(trans.obs)
            obs2_buf.append(trans.next_obs)
            act_buf.append(trans.action)
            rew_buf.append(trans.reward)
            done_buf.append(trans.done)
        batch_dict = dict(obs=obs_buf, obs2=obs2_buf, act=act_buf, rew=rew_buf, done=done_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch_dict.items()}


def sac(global_ac, global_ac_targ, rank, T, E, args, scores, wins,buffer, ac_kwargs=dict(), env=None, p2=None, seed=0,
        total_episode=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, min_alpha=0.2, fix_alpha=False, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, max_ep_len=1000, save_freq=1, device=None, tensorboard_dir=None, p2_list= None):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    # writer = GlobalSummaryWriter.getSummaryWriter()
    tensorboard_dir = os.path.join(tensorboard_dir, str(rank))
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    # env = gym.make(args.env)
    if not args.non_station:
        env = make_ftg_ram(args.env, p2=args.p2)
    else:
        env = make_ftg_ram_nonstation(args.env, p2_list=args.list, total_episode=args.station_rounds,stable=args.stable)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print("set up child process env")
    local_ac = MLPActorCritic(obs_dim, act_dim, **ac_kwargs).to(device)
    state_dict = global_ac.state_dict()
    local_ac.load_state_dict(state_dict)
    print("local ac load global ac")

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    # Async Version
    for p in global_ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=args.replay_size)
    # replay_buffer = ReplayBufferShare(buffer=buffer, size=args.replay_size)
    # if args.traj_dir:
    #     replay_buffer.store_trajectory(load_trajectory(args.traj_dir))

    # Entropy Tuning
    target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()  # heuristic value from the paper
    alpha = max(local_ac.log_alpha.exp().item(), min_alpha) if not fix_alpha else min_alpha

    # Set up optimizers for policy and q-function
    # Async Version
    pi_optimizer = Adam(global_ac.pi.parameters(), lr=lr, eps=1e-4)
    q1_optimizer = Adam(global_ac.q1.parameters(), lr=lr, eps=1e-4)
    q2_optimizer = Adam(global_ac.q2.parameters(), lr=lr, eps=1e-4)
    alpha_optim = Adam([global_ac.log_alpha], lr=lr, eps=1e-4)

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0
    trajectory = list()
    discard = False
    t = T.value()
    e = E.value()
    # Main loop: collect experience in env and update/log each epoch
    while e <= total_episode:

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = local_ac.get_action(o, device=device)
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
        trajectory.append(Transition(obs=o,action=a,reward=r,next_obs=o2,done=d))

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        T.increment()
        t = T.value()

        # End of trajectory handling
        if d or (ep_len == max_ep_len) or discard:
            E.increment()
            e = E.value()
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            if info.get('win', False):
                wins.append(1)
            else:
                wins.append(0)
            scores.append(ep_ret)
            m_score = np.mean(scores[-100:])
            win_rate = np.mean(wins[-100:])
            print(
                "Process {}, opponent:{}, # of global_episode :{},  # of global_steps :{}, round score: {}, mean score : {:.1f}, win_rate:{}, steps: {}, alpha: {}".format(
                    rank, p2, e, t, ep_ret, m_score, win_rate, ep_len, alpha))
            writer.add_scalar("metrics/round_score", ep_ret, e)
            writer.add_scalar("metrics/mean_score", m_score.item(), e)
            writer.add_scalar("metrics/win_rate", win_rate.item(), e)
            writer.add_scalar("metrics/round_step", ep_len, e)
            writer.add_scalar("metrics/alpha", alpha, e)
            o, ep_ret, ep_len = env.reset(), 0, 0
            trajectory = list()
            discard = False

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):

                batch = replay_buffer.sample_batch(batch_size,device=device)
                # First run one gradient descent step for Q1 and Q2
                q1_optimizer.zero_grad()
                q2_optimizer.zero_grad()
                loss_q = local_ac.compute_loss_q(batch, global_ac_targ, gamma, alpha)
                loss_q.backward()

                # Next run one gradient descent step for pi.
                pi_optimizer.zero_grad()
                loss_pi, entropy = local_ac.compute_loss_pi(batch, alpha)
                loss_pi.backward()

                alpha_optim.zero_grad()
                alpha_loss = -(local_ac.log_alpha * (entropy + target_entropy).detach()).mean()
                alpha_loss.backward(retain_graph=False)
                alpha = max(local_ac.log_alpha.exp().item(), min_alpha) if not fix_alpha else min_alpha

                nn.utils.clip_grad_norm_(local_ac.parameters(), 20)
                for global_param, local_param in zip(global_ac.parameters(), local_ac.parameters()):
                    global_param._grad = local_param.grad

                pi_optimizer.step()
                q1_optimizer.step()
                q2_optimizer.step()
                alpha_optim.step()

                state_dict = global_ac.state_dict()
                local_ac.load_state_dict(state_dict)

                # Finally, update target networks by polyak averaging.
                with torch.no_grad():
                    for p, p_targ in zip(global_ac.parameters(), global_ac_targ.parameters()):
                        p_targ.data.copy_((1 - polyak) * p.data + polyak * p_targ.data)

                writer.add_scalar("training/pi_loss", loss_pi.detach().item(), t)
                writer.add_scalar("training/q_loss", loss_q.detach().item(), t)
                writer.add_scalar("training/alpha_loss", alpha_loss.detach().item(), t)
                writer.add_scalar("training/entropy", entropy.detach().mean().item(), t)

        if t % save_freq == 0 and t > 0:
            torch.save(global_ac.state_dict(), os.path.join(args.save_dir, args.exp_name, args.model_para))
            state_dict_trans(global_ac.state_dict(), os.path.join(args.save_dir, args.exp_name,  args.numpy_para))
            torch.save((e, t, list(scores), list(wins)), os.path.join(args.save_dir, args.exp_name, args.train_indicator))
            print("Saving model at episode:{}".format(t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default="CartPole-v0")
    parser.add_argument('--env', type=str, default="FightingiceDataFrameskip-v0")
    parser.add_argument('--p2', type=str, default="ReiwaThunder")
    parser.add_argument('--non_station', default=False, action='store_true')
    parser.add_argument('--stable', default=False, action='store_true')
    parser.add_argument('--station_rounds', type=int, default=1000)
    parser.add_argument('--replay_size', type=int, default=10000)
    parser.add_argument('--list', nargs='+')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--update_after', type=int, default=10000)
    parser.add_argument('--min_alpha', type=float, default=0.3)
    parser.add_argument('--fix_alpha', default=False, action="store_true")
    parser.add_argument('--cuda',default=False, action='store_true')
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--episode', type=int, default=100000)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--n_process', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--save-dir', type=str, default="./experiments")
    parser.add_argument('--traj_dir', type=str, default="./experiments")
    parser.add_argument('--model_para', type=str, default="test.torch")
    parser.add_argument('--numpy_para', type=str, default="test.numpy")
    parser.add_argument('--train_indicator', type=str, default="test.data")
    args = parser.parse_args()

    # Basic Settings
    mp.set_start_method("forkserver")
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(torch.get_num_threads())
    experiment_dir = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    tensorboard_dir = os.path.join(experiment_dir, "runs")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    with open(os.path.join(experiment_dir, "arguments"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    # env and model setup
    ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)
    # env = gym.make(args.env)
    if not args.non_station:
        env = make_ftg_ram(args.env, p2=args.p2)
    else:
        env = make_ftg_ram_nonstation(args.env, p2_list=args.list, total_episode=args.station_rounds, stable=args.stable)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    global_ac = MLPActorCritic(obs_dim, act_dim, **ac_kwargs)

    # async training setup
    T = Counter()
    E = Counter()
    scores = mp.Manager().list()
    wins = mp.Manager().list()
    buffer = mp.Manager().list()

    if os.path.exists(os.path.join(args.save_dir, args.exp_name, args.model_para)):
        global_ac.load_state_dict(torch.load(os.path.join(args.save_dir, args.exp_name, args.model_para)))
        print("load model")
    if os.path.exists(os.path.join(args.save_dir, args.exp_name, args.train_indicator)):
        (e, t, scores_list, wins_list) = torch.load(os.path.join(args.save_dir, args.exp_name, args.train_indicator))
        T.set(t)
        E.set(e)
        scores.extend(scores_list)
        wins.extend(wins_list)
        print("load training indicator")

    global_ac_targ = deepcopy(global_ac)
    env.close()
    del env
    global_ac.share_memory()
    global_ac_targ.share_memory()
    if args.cuda:
        global_ac.to(device)
        global_ac_targ.to(device)
    var_counts = tuple(count_vars(module) for module in [global_ac.pi, global_ac.q1, global_ac.q2])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    processes = []
    for rank in range(args.n_process):  # + 1 for test process
        # if rank == 0:
            # p = mp.Process(target=test, args=(global_model,))
        # else:
        single_version_kwargs = dict(ac_kwargs=ac_kwargs, env=args.env, p2=args.p2,  p2_list=args.list, gamma=args.gamma, seed=args.seed,
                                     total_episode=args.episode, lr=args.lr, min_alpha=args.min_alpha, fix_alpha=True,
                                     update_after=args.update_after, batch_size=args.batch_size, start_steps=args.start_steps,
                                     replay_size=args.replay_size, update_every=1, max_ep_len=1000, save_freq=args.save_freq, polyak=0.995,
                                     device=device, tensorboard_dir=tensorboard_dir)
        p = mp.Process(target=sac, args=(global_ac, global_ac_targ, rank, T, E, args, scores, wins, buffer), kwargs=single_version_kwargs)
        p.start()
        time.sleep(5)
        processes.append(p)
    for p in processes:
        p.join()



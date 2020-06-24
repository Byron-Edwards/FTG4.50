import gym
import gym_fightingice
import random
import collections
import torch
import argparse
import copy
import os
import numpy as np
from time import sleep
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

# Characteristics
# 1. Discrete action space, single thread version.
# 2. Does not support trust-region updates.

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
buffer_limit = 10000
rollout_len = 600
batch_size = 32  # Indicates 4 sequences per mini-batch (4*rollout_len = 40 samples total)
c = 0.75  # For truncating importance sampling ratio

parser = argparse.ArgumentParser(description='ACER')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--cuda', type=bool, default=True, help='cuda Device')
parser.add_argument('--num-processes', type=int, default=4, metavar='N', help='Number of training async agents (does not include single validation agent)')
parser.add_argument('--T-max', type=int, default=10e7, metavar='STEPS', help='Number of training steps')
parser.add_argument('--t-max', type=int, default=300, metavar='STEPS', help='Max number of forward steps for A3C before update')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--hidden-size', type=int, default=32, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--model',default="",type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory',default="", type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--data',default="", type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--on-policy', action='store_true', help='Use pure on-policy training (A3C)')
parser.add_argument('--memory-capacity', type=int, default=100000, metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-ratio', type=int, default=4, metavar='r', help='Ratio of off-policy to on-policy updates')
parser.add_argument('--replay_start', type=int, default=1000, metavar='EPISODES', help='Number of transitions to save before starting off-policy training')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--trace-decay', type=float, default=1, metavar='λ', help='Eligibility trace decay factor')
parser.add_argument('--trace-max', type=float, default=10, metavar='c', help='Importance weight truncation (max) value')
parser.add_argument('--trust-region', action='store_true', help='Use trust region')
parser.add_argument('--trust-region-decay', type=float, default=0.99, metavar='α', help='Average model weight decay rate')
parser.add_argument('--trust-region-threshold', type=float, default=1, metavar='δ', help='Trust region threshold value')
parser.add_argument('--reward-clip', action='store_true', help='Clip rewards to [-1, 1]')
parser.add_argument('--lr', type=float, default=0.0001, metavar='η', help='Learning rate')
parser.add_argument('--lr-deca y', action='store_true', help='Linearly decay learning rate to 0')
parser.add_argument('--rmsprop-decay', type=float, default=0.99, metavar='α', help='RMSprop decay factor')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Off-policy batch size')
parser.add_argument('--entropy-weight', type=float, default=0.0001, metavar='β', help='Entropy regularisation weight')
parser.add_argument('--max-gradient-norm', type=float, default=40, metavar='VALUE', help='Gradient L2 normalisation')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=25000, metavar='STEPS', help='Number of training steps between evaluations (roughly)')
parser.add_argument('--evaluation-episodes', type=int, default=20, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--render', action='store_true', help='Render evaluation agent')
parser.add_argument('--name', type=str, default='./OpenAI/ACER', help='Save folder')
parser.add_argument('--env', type=str, default='FightingiceDataNoFrameskip-v0',help='environment name')
parser.add_argument('--port', type=int, default=5000,help='FightingICE running Port')
parser.add_argument('--p2', type=str, default="RHEA_PI",help='FightingICE running Port')


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, seq_data):
        self.buffer.append(seq_data)

    def sample(self, on_policy=False):
        if on_policy:
            mini_batch = [self.buffer[-1]]
        else:
            mini_batch = random.sample(self.buffer, batch_size)

        s_lst, a_lst, r_lst, prob_lst, done_lst, is_first_lst = [], [], [], [], [], []
        for seq in mini_batch:
            is_first = True  # Flag for indicating whether the transition is the first item from a sequence
            for transition in seq:
                s, a, r, prob, done = transition

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r)
                prob_lst.append(prob)
                done_mask = 0.0 if done else 1.0
                done_lst.append(done_mask)
                is_first_lst.append(is_first)
                is_first = False

        s, a, r, prob, done_mask, is_first = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                             torch.tensor(r_lst), torch.tensor(prob_lst, dtype=torch.float), torch.tensor(done_lst), \
                                             torch.tensor(is_first_lst)
        return s, a, r, prob, done_mask, is_first

    def size(self):
        return len(self.buffer)


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size):
        super(ActorCritic, self).__init__()
        self.state_size = observation_space.shape[0]
        self.action_size = action_space.n

        self.fc1 = nn.Linear(self.state_size, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.fc_actor1 = nn.Linear(hidden_size, hidden_size)
        self.fc_critic1 = nn.Linear(hidden_size, hidden_size)
        self.fc_critic = nn.Linear(hidden_size, self.action_size)
        self.fc_actor = nn.Linear(hidden_size, self.action_size)
        self.hx = torch.zeros(1, args.hidden_size)
        self.cx = torch.zeros(1, args.hidden_size)

    def pi(self, x,softmax_dim=0):
            x = F.relu(self.fc1(x))
            h = self.lstm(x, (self.hx, self.cx))  # h is (hidden state, cell state)
            x = h[0]
            policy = F.softmax(self.fc_actor(F.relu(self.fc_actor1(x))), dim=1).clamp(
                max=1 - 1e-20)  # Prevent 1s and hence NaNs
            return policy

    def q(self, x):
            x = F.relu(self.fc1(x))
            h = self.lstm(x, (self.hx, self.cx))  # h is (hidden state, cell state)
            x = h[0]
            Q = self.fc_critic(F.relu(self.fc_critic1(x)))
            return Q


  # def forward(self, x, h):
  #   x = F.relu(self.fc1(x))
  #   h = self.lstm(x, h)  # h is (hidden state, cell state)
  #   x = h[0]
  #   policy = F.softmax(self.fc_actor(F.relu(self.fc_actor1(x))), dim=1).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
  #   Q = self.fc_critic(F.relu(self.fc_critic1(x)))
  #   V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π
  #   return policy, Q, V, h


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


def train(model, optimizer, memory, on_policy=False, device=torch.device("cuda")):
    s, a, r, prob, done_mask, is_first = memory.sample(on_policy)
    s = s.to(device)
    a = a.to(device)
    r = r.to(device)
    prob = prob.to(device)
    done_mask = done_mask.to(device)
    is_first = is_first.to(device)
    q = model.q(s)
    q_a = q.gather(1, a)
    pi = model.pi(s, softmax_dim=1)
    pi_a = pi.gather(1, a)
    v = (q * pi).sum(1).unsqueeze(1).detach()

    rho = (pi.detach() / prob)
    rho_a = rho.gather(1, a)
    rho_bar = rho_a.clamp(max=c)
    correction_coeff = ((1 - c / rho).clamp(min=0))

    q_ret = (v[-1] * done_mask[-1])
    q_ret_lst = []
    for i in reversed(range(len(r))):
        q_ret = r[i] + gamma * q_ret
        q_ret_lst.append(q_ret.item())
        q_ret = rho_bar[i] * (q_ret - q_a[i]) + v[i]

        if is_first[i] and i != 0:
            q_ret = v[i - 1] * done_mask[i - 1]  # When a new sequence begins, q_ret is initialized

    q_ret_lst.reverse()
    q_ret = torch.tensor(q_ret_lst, dtype=torch.float).unsqueeze(1).to(device)

    loss1 = -rho_bar * torch.log(pi_a) * (q_ret - v)
    loss2 = -correction_coeff * pi.detach() * torch.log(pi) * (q.detach() - v)  # bias correction term
    loss = loss1 + loss2.sum(1) + F.smooth_l1_loss(q_a, q_ret)

    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()


def actor(rank, args, T, BEST,memory_queue,model_queue):
    torch.manual_seed(args.seed + rank)
    env = gym.make(args.env, java_env_path=".", port=args.port + rank * 2)
    env.seed(args.seed + rank)
    model = ActorCritic(env.observation_space, env.action_space)

    scores = []
    n_epi = 0
    action_entropy = 4.025

    # Actor Loop
    while T.value() <= args.T_max:
        s = env.reset(p2=args.p2)
        if not model_queue.empty():
            model.load_state_dict(model_queue.get())
            print("Load New Model at EPISODE {}".format(T.value()))
        done = False
        discard = False
        round_score = 0
        while not done:

            seq_data = []
            for t in range(rollout_len):
                if isinstance(s, list):
                    s = s[0]
                prob = model.pi(torch.from_numpy(s).float())
                action_entropy = Categorical(probs=prob).entropy()
                a = Categorical(prob).sample().item()
                s_prime, r, done, info = env.step(a)
                if info.get('no_data_receive', False):
                    env.close()
                    discard = True
                seq_data.append((s, a, r, prob.detach().cpu().numpy(), done))

                round_score += r
                s = s_prime
                if done or discard:
                    break
            if not discard:
                memory_queue.put(seq_data)
                scores.append(round_score)
                print("PUT DATA")
            else:
                break
        if not discard:
            n_epi += 1
            T.increment()
            m_score = np.mean(scores[-50:])
            print("Process: {}, # of episode :{}, round score : {}, 100 round mean score: {}".format(rank, n_epi, round_score, m_score))
            if m_score*400 > BEST.value():
                BEST.set(int(m_score*400))
    env.close()


if __name__ == '__main__':
    # BLAS setup
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # mp.set_start_method("spawn")
    # Setup
    args = parser.parse_args()
    # Creating directories.
    save_dir = os.path.join(args.name, 'checkpoint')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(' ' * 26 + 'Options')

    # Saving parameters
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for k, v in vars(args).items():
            print(' ' * 26 + k + ': ' + str(v))
            f.write(k + ' : ' + str(v) + '\n')

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    gym.logger.set_level(gym.logger.INFO)  # Disable Gym warnings

    T = Counter()  # Global shared counter
    BEST = Counter()
    BEST.set(-9999)
    pre_best = BEST.value()
    # writer = SummaryWriter(log_dir=save_dir, comment="-" + args.env + "-" + args.p2)
    memory = ReplayBuffer()
    env = gym.make(args.env, java_env_path=".", port=args.port)
    model = ActorCritic(env.observation_space, env.action_space, )
    shared_model = copy.deepcopy(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    env.close()
    del env

    if args.model and os.path.isfile(args.model):
        # Load pretrained weights
        print("Load model from checkpoint {}".format(args.model))
        model.load_state_dict(torch.load(args.model))
        shared_model.load_state_dict(torch.load(args.model, map_location="cpu"))
    if args.memory and os.path.isfile(args.memory):
        memory = torch.load(args.memory)
        print("Load memory from CheckPoint {}, memory len: {}".format(args.data, len(memory)))
    if args.data and os.path.isfile(args.data):
        T.set(torch.load(args.data)[0])
        BEST.set(torch.load(args.data)[1])
        pre_best = BEST.value()
        print("Load data from CheckPoint {}, T: {}.BEST: {}".format(args.data, T.value(),BEST.value()))

    memory_queue = mp.Queue()
    model_queue = mp.Queue()
    processes = []
    if not args.evaluate:
        # Start training agents
        for rank in range(1, args.num_processes + 1):
            model_queue.put(shared_model.state_dict())
            p = mp.Process(target=actor, args=(rank, args, T, BEST, memory_queue,model_queue))
            p.start()
            sleep(15)
            print('Process ' + str(rank) + ' started')
            processes.append(p)

    c_t = 0
    # Learner Loop
    while T.value() <= args.T_max:
        t = T.value()
        best = BEST.value()

        # receive data from actor then train and record into tensorboard
        if not memory_queue.empty():
            print("EPISODE: {}, BEST: {}".format(t, best))
            # writer.add_scalar("reward/round_score", round_score, T.value())
            # writer.add_scalar("reward/mean_score", m_score, T.value())
            # writer.add_scalar("indicator/entropy", action_entropy, T.value())
            memory.put(memory_queue.get())
            train(model, optimizer, memory, on_policy=True, device=device)
            print("Training Model on-policy at EPISODE {}".format(t))
            if memory.size() > args.replay_start:
                print("Training Model off-policy at EPISODE {}".format(t))
                train(model, optimizer, memory, device=device)

        # save the best model
        if best > pre_best:
            print("Save BEST model!")
            torch.save(model.state_dict(), os.path.join("OpenAI/ACER/checkpoint/", 'model_{}_{}'.format("BEST", best)))  # Save model params
            torch.save(memory, os.path.join("OpenAI/ACER/checkpoint/", 'memory_{}_{}'.format("BEST", best)))  # Save memory
            torch.save((t, best), os.path.join("OpenAI/ACER/checkpoint/", 'indicator_{}_{}'.format("BEST", best)))  # Save data
            pre_best = BEST.value()

        # deliver model from learner to actor and save the latest model
        if t % args.num_processes * 4 == 0 and t > 1 and c_t < t:
            shared_model = copy.deepcopy(model)
            shared_model.to(torch.device("cpu"))
            print("deep copy share model!")
            c_t = t
            for _ in range(args.num_processes):
                model_queue.put(shared_model.state_dict())
            print("Save LATEST model!")
            torch.save(model.state_dict(), os.path.join("OpenAI/ACER/checkpoint/", 'model_{}_{}'.format("LATEST", t)))  # Save model params
            torch.save(memory, os.path.join("OpenAI/ACER/checkpoint/", 'memory_{}_{}'.format("LATEST", t)))  # Save memory
            torch.save((t, best), os.path.join("OpenAI/ACER/checkpoint/", 'indicator_{}_{}'.format("LATEST", t)))  # Save data

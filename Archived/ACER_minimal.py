import random
import torch
import argparse
import copy
import math
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import itertools
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from OppModeling.atari_wrappers import *
# Characteristics
# 1. Discrete action space, single thread version.
# 2. Does not support trust-region updates.

# Hyper parameters
parser = argparse.ArgumentParser(description='ACER')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--cuda', type=bool, default=True, help='cuda Device')
parser.add_argument('--num-processes', type=int, default=1, metavar='N', help='Number of training async agents (does not include single validation agent)')
parser.add_argument('--T-max', type=int, default=10e7, metavar='STEPS', help='Number of training steps')
parser.add_argument('--t-max', type=int, default=300, metavar='STEPS', help='Max number of forward steps for A3C before update')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--hidden-size', type=int, default=32, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--model',default="",type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory',default="", type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--data',default="", type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--on-policy', action='store_true', help='Use pure on-policy training (A3C)')
parser.add_argument('--memory-capacity', type=int, default=5000, metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-ratio', type=int, default=4, metavar='r', help='Ratio of off-policy to on-policy updates')
parser.add_argument('--replay_start', type=int, default=5000, metavar='EPISODES', help='Number of transitions to save before starting off-policy training')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--trace-decay', type=float, default=1, metavar='λ', help='Eligibility trace decay factor')
parser.add_argument('--trace-max', type=float, default=10, metavar='c', help='Importance weight truncation (max) value')
parser.add_argument('--trust-region', default=False, action='store_true', help='Use trust region')
parser.add_argument('--trust-region-decay', type=float, default=0.99, metavar='α',
                    help='Average model weight decay rate')
parser.add_argument('--trust-region-threshold', type=float, default=1, metavar='δ', help='Trust region threshold value')
parser.add_argument('--reward-clip', action='store_true', help='Clip rewards to [-1, 1]')
parser.add_argument('--lr', type=float, default=0.0001, metavar='η', help='Learning rate')
parser.add_argument('--lr-decay', default=True, action='store_true', help='Linearly decay learning rate to 0')
parser.add_argument('--lr-min', default=1e-6, type=float, help='minimal learning rate')
parser.add_argument('--rmsprop-decay', type=float, default=0.99, metavar='α', help='RMSprop decay factor')
parser.add_argument('--batch-size', type=int, default=8, metavar='SIZE', help='Off-policy batch size')
parser.add_argument('--entropy-weight', type=float, default=0.01, metavar='β', help='Entropy regularisation weight')
parser.add_argument('--max-gradient-norm', type=float, default=10, metavar='VALUE', help='Gradient L2 normalisation')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=25000, metavar='STEPS', help='Number of training steps between evaluations (roughly)')
parser.add_argument('--evaluation-episodes', type=int, default=20, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--render', action='store_true', help='Render evaluation agent')
parser.add_argument('--name', type=str, default='./OpenAI/ACER', help='Save folder')
parser.add_argument('--env', type=str, default='FightingiceDataNoFrameskip-v0',help='environment name')
parser.add_argument('--port', type=int, default=5000,help='FightingICE running Port')
parser.add_argument('--p2', type=str, default="Toothless",help='FightingICE running Port')
args = parser.parse_args()


class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen=args.memory_capacity)
        self.checkpoint = 0

    def put(self, seq_data):
        self.buffer.append(seq_data)
        if self.size() == args.memory_capacity:
            self.checkpoint -= 1

    def sample(self, on_policy=False):
        if on_policy:
            mini_batch = [self.buffer[-1]]
        else:
            mini_batch = random.sample(self.buffer, args.batch_size)

        s_lst, a_lst, r_lst, prob_lst, done_lst, is_first_lst, action_mask_list = [], [], [], [], [], [],[]
        for seq in mini_batch:
            is_first = True  # Flag for indicating whether the transition is the first item from a sequence
            for transition in seq:
                s, a, r, prob, done, action_mask = transition

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r)
                prob_lst.append(prob)
                done_mask = 0.0 if done else 1.0
                done_lst.append(done_mask)
                is_first_lst.append(is_first)
                action_mask_list.append(action_mask)
                is_first = False

        s, a, r, prob, done_mask, is_first, action_mask = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst),\
                                                          torch.tensor(r_lst), torch.tensor(prob_lst, dtype=torch.float),\
                                                          torch.tensor(done_lst), torch.tensor(is_first_lst), \
                                                          torch.tensor(action_mask_list)
        return s, a, r, prob, done_mask, is_first, action_mask

    def size(self):
        return len(self.buffer)

    def save(self, save_dir, save_all=False, save_all_interval=500):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            print("Create memory saving directory at {}".format(save_dir))
        if save_all:
            for i in range(0, self.size(), save_all_interval):
                if i + save_all_interval > self.size():
                    deque_slice = deque(itertools.islice(self.buffer, i, self.size()))
                else:
                    deque_slice = deque(itertools.islice(self.buffer, i, i + save_all_interval))
                torch.save(deque_slice,
                           os.path.join(save_dir, 'memory_{}_{}_{}'.format(i, i + len(deque_slice),
                                                                           datetime.now().strftime("%Y%m%d-%H%M%S"))))
        else:
            if self.checkpoint == self.size():
                return
            deque_slice = deque(itertools.islice(self.buffer, self.checkpoint, self.size()))
            torch.save(deque_slice,
                   os.path.join(save_dir, 'memory_{}_{}_{}'.format(self.checkpoint, self.size(),
                                                                   datetime.now().strftime("%Y%m%d-%H%M%S"))))
        self.checkpoint = self.size()

    def load(self, save_dir):
        for filename in os.listdir(save_dir):
            memory_sequence = os.path.join(save_dir, filename)
            self.buffer.extend(torch.load(memory_sequence))
        self.checkpoint = self.size()


class ActorCritic(nn.Module):
    def __init__(self,observation_space, action_space, hidden_size):
        super(ActorCritic, self).__init__()
        self.state_size = observation_space
        self.action_size = action_space.n
        self.fc1 = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.LeakyReLU(),
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc_pi = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, self.action_size),
        )
        self.fc_q = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, self.action_size),
        )

    def pi(self, x, action_mask, softmax_dim=0):
        x = self.fc1(x)
        x = self.fc_pi(x).masked_fill(action_mask, float("-inf"))
        pi = F.softmax(x, dim=softmax_dim).clamp(min=0 + 1e-20, max=1 - 1e-20)
        return pi

    def q(self, x):
        x = self.fc1(x)
        q = self.fc_q(x)
        return q


# Knuth's algorithm for generating Poisson samples
def _poisson(lmbd):
    L, k, p = math.exp(-lmbd), 0, 1
    while p > L:
        k += 1
        p *= random.uniform(0, 1)
    return max(k - 1, 0)


class TimeoutError(Exception): pass


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


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


def flip_obs(s):
    my_info = s[0:74]
    opp_info = s[74:148]
    game_frame = s[148]
    my_attack = s[149:199]
    opp_attack = s[199:249]
    flip_s = opp_info + my_info + game_frame + opp_attack + my_attack
    return flip_s


def obs_get_action(s):
    my_action = s[14:70]
    for i in range(len(my_action)):
        if my_action[i] == 1:
            return i


def train(model, average_model, t, optimizer, memory, on_policy=False, device=torch.device("cuda")):
    print("Training {}".format("on-policy" if on_policy else "off-policy"))
    s, a, r, prob, done_mask, is_first, action_mask = memory.sample(on_policy)
    s = s.to(device)
    a = a.to(device)
    r = r.to(device)
    prob = prob.to(device)
    done_mask = done_mask.to(device)
    is_first = is_first.to(device)
    action_mask = action_mask.to(device)
    q = model.q(s)
    q_a = q.gather(1, a)
    pi = model.pi(s, action_mask, softmax_dim=1)
    pi_a = pi.gather(1, a)
    v = (q * pi).sum(1).unsqueeze(1).detach()

    if args.trust_region:
        avg_pi = average_model.pi(s, action_mask, softmax_dim=1)
        avg_pi_a = avg_pi.gather(1, a)

    rho = (pi.detach() / prob)
    rho_a = rho.gather(1, a)
    rho_bar = rho_a.clamp(max=args.trace_max)
    correction_coeff = ((1 - args.trace_max / rho).clamp(min=0))

    q_ret = (v[-1] * done_mask[-1])
    q_ret_lst = []
    for i in reversed(range(len(r))):
        q_ret = r[i] + args.discount * q_ret
        q_ret_lst.append(q_ret.item())
        q_ret = rho_bar[i] * (q_ret - q_a[i]) + v[i]

        if is_first[i] and i != 0:
            q_ret = v[i - 1] * done_mask[i - 1]  # When a new sequence begins, q_ret is initialized

    q_ret_lst.reverse()
    q_ret = torch.tensor(q_ret_lst, dtype=torch.float).unsqueeze(1).to(device)

    loss1 = -(rho_bar * (q_ret - v)).detach() * pi_a.log()
    loss2 = -pi.log() * (correction_coeff * pi * (q - v)).detach()  # bias correction term
    entropy_reg = args.entropy_weight * -(torch.log(pi) * pi).sum(1)
    policy_loss = loss1 + loss2.sum(1) - entropy_reg
    value_loss = F.smooth_l1_loss(q_a, q_ret.detach())
    optimizer.zero_grad()

    if args.trust_region:
        # KL divergence k ← ∇θ0∙DKL[π(∙|s_i; θ_a) || π(∙|s_i; θ)]
        k = -avg_pi / pi
        g = torch.autograd.grad(-policy_loss.mean(), pi)[0]
        # g = (rho_bar * (q_ret - v) / pi_a + (correction_coeff * pi * (q_a - v) / pi).sum(1)).detach()
        # Policy update dθ ← dθ + ∂θ/∂θ∙z*
        # kl = - (avg_pi_a * (pi_a.log() - avg_pi_a.log())).sum(1).mean(0)
        # Compute dot products of gradients
        k_dot_g = (k * g).sum(1).mean(0)
        k_dot_k = (k * k).sum(1).mean(0)
        # Compute trust region update
        trust_factor = ((k_dot_g - args.trust_region_threshold) / k_dot_k).clamp(min=0)
        g = g - trust_factor * k
        # z* = g - max(0, (k^T∙g - δ) / ||k||^2_2)∙k
        grads_f = -g
        torch.autograd.backward(pi, grad_tensors=(grads_f,), retain_graph=True)
        value_loss.backward()
    else:
        loss = policy_loss + value_loss
        loss.mean().backward()

    nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
    optimizer.step()

    lr = args.lr
    if args.lr_decay:
        # Linearly decay learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(args.lr
                                    # * (0.8 ** (n_bounce + 1))
                                    * (0.5**((t-t_bounce)/2000)), args.lr_min)
            lr = param_group['lr']

    if args.trust_region:
        for model_param, average_param in zip(model.parameters(), average_model.parameters()):
            average_param.data = args.trust_region_decay * average_param.data + (1 - args.trust_region_decay) * model_param.data

    writer.add_scalar("loss/policy_loss", policy_loss.mean().detach(), t)
    writer.add_scalar("loss/value_loss", value_loss.mean().detach(), t)
    writer.add_scalar("loss/entropy_reg", entropy_reg.mean().detach(), t)
    writer.add_scalar("loss/learning_rate", lr, t)


def actor(rank, args, T,memory_queue,model_queue,p2):
    torch.manual_seed(args.seed + rank)
    # env = FrameStack(gym.make(args.env), 4)
    env = gym.make(args.env, java_env_path="..", port=args.port + rank * 2, p2=p2)
    print("Process {} fighting with {}".format(rank, p2))
    env.seed(args.seed + rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space,args.hidden_size)
    n_epi = 0

    # Actor Loop
    while T.value() <= args.T_max:
        t_value = T.value()
        try:
            with timeout(seconds=30):
                s = env.reset(p2=p2)
                # opp_s = flip_obs(s)
        except TimeoutError:
            print("Time out to reset env")
            env.close()
            continue
        if not model_queue.empty():
            print("Process {} going to load new model at EPISODE {}......".format(rank, t_value))
            received_obj = model_queue.get()
            model_dict = copy.deepcopy(received_obj)
            model.load_state_dict(model_dict)
            print("Process {} finished loading new mode at EPISODE {}!!!!!!".format(rank, t_value))
            del received_obj
        action_mask = [False for _ in range(env.action_space.n)]
        action_mask = torch.BoolTensor(action_mask)
        done = False
        discard = False
        round_score = 0
        episode_length = 0
        sum_entropy = 0
        seq_data = []
        while not done:
            env.render()
            prob = model.pi(torch.from_numpy(s).float(), action_mask)
            sum_entropy += Categorical(probs=prob.detach()).entropy()
            a = Categorical(prob.detach()).sample().item()
            s_prime, r, done, info = env.step(a)
            # (opp_s_prime, opp_r, opp_done, _) = info.get('opp_transit', False)
            # opp_a = obs_get_action(opp_s_prime)
            if info.get('no_data_receive', False):
                env.close()
                discard = True
                break
            valid_actions = info.get('my_action_enough', {})
            # get valid actions
            if len(valid_actions) > 0:
                action_mask = [False if i in valid_actions else True for i in range(56)]
            else:
                action_mask = [False for _ in range(env.action_space.n)]
            action_mask = torch.BoolTensor(action_mask)
            seq_data.append((s, a, r, prob.detach().numpy(), done, action_mask.detach().numpy()))
            round_score += r
            s = s_prime
            episode_length += 1
        if not discard:
            n_epi += 1
            on_policy_data = (seq_data, (episode_length, round_score, sum_entropy / episode_length))
            print("Process: {}, # of episode :{}, round score : {}, episode_length: {}".format(rank, n_epi, round_score, episode_length))
            send_object = copy.deepcopy(on_policy_data)
            memory_queue.put(send_object, )
            print("Process {} send trajectory".format(rank))
    env.close()


if __name__ == '__main__':
    # BLAS setup
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # mp.set_start_method("spawn")
    # Setup
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

    tensorboard_dir = os.path.join(save_dir, "runs",datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    gym.logger.set_level(gym.logger.INFO)  # Disable Gym warnings

    T = Counter()  # Global shared counter
    BEST = Counter()
    BEST.set(-9999)
    pre_best = BEST.value()
    # writer = SummaryWriter(log_dir=save_dir, comment="-" + args.env + "-" + args.p2)
    memory = ReplayBuffer()
    writer = SummaryWriter(log_dir=tensorboard_dir)
    # env = make_env(args.env)
    env = gym.make(args.env, java_env_path="..", port=args.port, p2=args.p2)
    model = ActorCritic(env.observation_space.shape[0], env.action_space, args.hidden_size)
    shared_model = copy.deepcopy(model)
    model.to(device)
    if args.trust_region:
        average_model = copy.deepcopy(model)
        average_model.to(device)
    else:
        average_model = None
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scores = []
    m_scores = []
    env.close()
    del env

    if args.model and os.path.isfile(args.model):
        # Load pretrained weights
        print("Load model from checkpoint {}".format(args.model))
        model.load_state_dict(torch.load(args.model))
        shared_model.load_state_dict(torch.load(args.model, map_location="cpu"))
    if args.memory and os.path.isdir(args.memory):
        memory.load(args.memory)
        print("Load memory from CheckPoint {}, memory len: {}".format(args.memory, memory.size()))
    if args.data and os.path.isfile(args.data):
        T.set(torch.load(args.data)[0])
        BEST.set(torch.load(args.data)[1])
        scores = torch.load(args.data)[2]
        m_scores = torch.load(args.data)[3]
        pre_best = BEST.value()
        print("Load data from CheckPoint {}, T: {}.BEST: {}".format(args.data, T.value(), BEST.value()))

    memory_queue = mp.SimpleQueue()
    model_queue = mp.SimpleQueue()
    processes = []
    p2_list = ["ReiwaThunder", "RHEA_PI", "Toothless", "FalzAI"]
    if not args.evaluate:
        # Start training agents
        for rank in range(1, args.num_processes + 1):
            p2 = p2_list[(rank-1) % len(p2_list)]
            model_queue.put(shared_model.state_dict())
            p = mp.Process(target=actor, args=(rank, args, T, memory_queue, model_queue, args.p2))
            p.start()
            # sleep(15)
            print('Process ' + str(rank) + ' started')
            processes.append(p)

    c_t = 0

    t_bounce, n_bounce = 0,0
    # Learner Loop
    while T.value() <= args.T_max:
        # receive data from actor then train and record into tensorboard
        print("Going to read data from ACTOR......")
        received_obj = memory_queue.get()
        print("Finish Reading data from ACTOR!!!!!!")
        on_policy_data = copy.deepcopy(received_obj)
        del received_obj

        T.increment()
        t = T.value()
        best = BEST.value()
        (trajectory, (episode_length, round_score, average_entropy)) = on_policy_data
        memory.put(trajectory)
        scores.append(round_score)
        m_score = np.mean(scores[-100:])
        m_scores.append(m_score)
        if m_score * 400 > BEST.value() and len(scores) >= args.replay_start:
            BEST.set(int(m_score * 400))
            best = BEST.value()
        writer.add_scalar("reward/round_score", round_score, t)
        writer.add_scalar("reward/mean_score", m_score, t)
        writer.add_scalar("indicator/entropy", average_entropy, t)
        writer.add_scalar("indicator/episode_length", episode_length, t)
        print("EPISODE: {}, BEST: {}, MEAN_SCORE: {}".format(t, best, m_score))
        train(model, average_model, t, optimizer, memory, on_policy=True, device=device)
        if memory.size() >= args.replay_start:
            # for _ in range(_poisson(args.replay_ratio)):
            train(model, average_model, t, optimizer, memory, device=device)

        # save the best model
        if best > pre_best:
            print("Save BEST model!")
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'model_{}'.format("BEST")))  # Save model params
            memory.save(os.path.join(save_dir,"memory"))
            torch.save((t, best, scores, m_scores),
                       os.path.join(save_dir, 'indicator_{}'.format("BEST")))  # Save data
            pre_best = best

        # deliver model from learner to actor and save the latest model
        if t % (args.num_processes * 10) == 0 and t > 1 and c_t < t:
            shared_model = copy.deepcopy(model)
            shared_model.to(torch.device("cpu"))
            shared_model_dict = copy.deepcopy(shared_model.state_dict())
            c_t = t
            for _ in range(args.num_processes):
                model_queue.put(shared_model_dict,)
            print("Saving LATEST model......")
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'model_{}'.format("LATEST",)))  # Save model params
            memory.save(os.path.join(save_dir,"memory"))
            torch.save((t, best, scores, m_scores),
                       os.path.join(save_dir, 'indicator_{}'.format("LATEST",)))  # Save data
            print("Save LATEST model!!!!!!")

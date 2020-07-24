import gym
import gym_fightingice
import torch
import os
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from OpenAI.atari_wrappers import make_ftg_ram
from gym_fightingice.envs.Machete import Machete

# Hyperparameters
n_train_processes = 8
save_interval = 20
save_dir = "./OpenAI/A3C"
learning_rate = 0.0002
update_interval = 5
gamma = 0.98
hidden_size = 256
entropy_weight = 0.01
env_name = "FightingiceDataFrameskip-v0"
p2 = "ReiwaThunder"


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


class ActorCritic(nn.Module):
    def __init__(self, state_n, action_n, hidden_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_n, hidden_size)
        self.fc_pi = nn.Linear(hidden_size, action_n)
        self.fc_v = nn.Linear(hidden_size, 1)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim).clamp(min=1e-20, max=1-1e-20)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


def train(global_model, rank, T, scores):
    env = make_ftg_ram(env_name, p2=p2)
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n
    local_model = ActorCritic(state_shape, action_shape, hidden_size)
    local_model.load_state_dict(global_model.state_dict())
    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)

    while True:
        discard = False
        done = False
        s = env.reset()
        score = 0
        sum_entropy = 0
        step = 0
        while not done:
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):
                prob = local_model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                if info.get('no_data_receive', False):
                    discard = True
                    break

                s_lst.append(s)
                a_lst.append([a])
                # r_lst.append(r/100.0)
                r_lst.append(r)

                s = s_prime

                score += r
                sum_entropy += Categorical(probs=prob.detach()).entropy()
                step += 1
                if done:
                    break
            if discard:
                break
            s_final = torch.tensor(s_prime, dtype=torch.float)
            R = 0.0 if done else local_model.v(s_final).item()
            td_target_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + reward
                td_target_lst.append([R])
            td_target_lst.reverse()

            s_batch, a_batch, td_target = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                torch.tensor(td_target_lst)
            advantage = td_target - local_model.v(s_batch)

            pi = local_model.pi(s_batch, softmax_dim=1)
            pi_a = pi.gather(1, a_batch)
            loss = -torch.log(pi_a) * advantage.detach() + \
                F.smooth_l1_loss(local_model.v(s_batch), td_target.detach()) - \
                   (entropy_weight * -(torch.log(pi) * pi).sum())

            optimizer.zero_grad()
            loss.mean().backward()
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())
        if discard:
            continue
        T.increment()
        t = T.value()
        scores.append(score)
        m_score = np.mean(scores[-100:])
        print("Process {}, # of episode :{}, round score: {}, mean score : {:.1f}, entropy: {}, steps: {}".format(rank, t, score, m_score, sum_entropy/step, step))
        if t % save_interval == 0 and t > 0:
            torch.save(global_model.state_dict(), os.path.join(save_dir, "model"))
            print("Saving model at episode:{}".format(t))



    # env.close()
    # print("Training process {} reached maximum episode.".format(rank))


def test(global_model):
    env = make_ftg_ram(env_name, p2=p2)
    score = 0.0
    print_interval = 20
    n_epi = 0
    while True:
        n_epi += 1
        done = False
        s = env.reset()
        while not done:
            prob = global_model.pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().item()
            s_prime, r, done, info = env.step(a)
            s = s_prime
            score += r

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(
                n_epi, score/print_interval))
            score = 0.0


if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    env = make_ftg_ram(env_name, p2=p2)
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n
    global_model = ActorCritic(state_shape, action_shape, hidden_size)
    if os.path.exists(os.path.join(save_dir, "model")):
        global_model.load_state_dict(torch.load(os.path.join(save_dir, "model")))
        print("load model")
    global_model.share_memory()
    T = Counter()
    scores = mp.Manager().list()
    env.close()
    del env

    processes = []
    for rank in range(n_train_processes ):  # + 1 for test process
        # if rank == 0:
            # p = mp.Process(target=test, args=(global_model,))
        # else:
        p = mp.Process(target=train, args=(global_model, rank, T, scores))
        p.start()
        time.sleep(15)
        processes.append(p)
    for p in processes:
        p.join()

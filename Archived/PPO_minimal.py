# PPO-LSTM
import gym
import gym_fightingice
import signal
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 2
T_horizon = 400
hidden_size = 128
save_dir = './OpenAI/PPO'
env_name = "FightingiceDataNoFrameskip-v0"
port = 4000
p2 = "RHEA_PI"
model_checkpoint = ""
data_checkpoint = ""


class PPO(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size):
        super(PPO, self).__init__()
        self.data = []
        self.hidden_size = hidden_size
        self.state_size = observation_space.shape[0]
        self.action_size = action_space.n

        self.fc1 = nn.Linear(self.state_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc_pi = nn.Linear(hidden_size, self.action_size)
        self.fc_v = nn.Linear(hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, self.hidden_size)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden

    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, self.hidden_size)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()


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


def main():
    env = gym.make(env_name, java_env_path="..", port=port, p2=p2)
    tensorboard_dir = os.path.join(save_dir, 'checkpoint', "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    model = PPO(env.observation_space, env.action_space, hidden_size)
    scores = []
    n_epi = 0
    m_scores = []
    pre_best = -9999

    if model_checkpoint and os.path.isfile(model_checkpoint):
        # Load pretrained weights
        print("Load model from checkpoint {}".format(model_checkpoint))
        model.load_state_dict(torch.load(model_checkpoint))
    if data_checkpoint and os.path.isfile(data_checkpoint):
        (n_epi, pre_best, scores, m_scores) = torch.load(data_checkpoint)
        print("Load data from CheckPoint {}, T: {}.BEST: {}".format(data_checkpoint, n_epi, pre_best))

    while True:
        score = 0.0
        step = 0
        sum_entropy = 0
        h_out = (torch.zeros([1, 1, hidden_size], dtype=torch.float), torch.zeros([1, 1, hidden_size], dtype=torch.float))
        try:
            with timeout(seconds=30):
                s = env.reset()
        except TimeoutError:
            print("Time out to reset env")
            env.close()
            continue
        done = False
        discard = False
        while not done:
            for t in range(T_horizon):
                h_in = h_out
                prob, h_out = model.pi(torch.from_numpy(s).float(), h_in)
                sum_entropy += Categorical(probs=prob.detach()).entropy()
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                if info.get('no_data_receive', False):
                    env.close()
                    discard = True
                    break
                model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), h_in, h_out, done))
                s = s_prime
                score += r
                step += 1
                if done:
                    break
            if discard:
                break
            model.train_net()
        if discard:
            continue

        n_epi += 1
        scores.append(score)
        m_score = np.mean(scores[-100:])
        m_scores.append(m_score)
        writer.add_scalar("reward/round_score", score, n_epi)
        writer.add_scalar("reward/mean_score", m_score, n_epi)
        writer.add_scalar("indicator/entropy", sum_entropy / step, n_epi)
        writer.add_scalar("indicator/episode_length", step, n_epi)
        # save the best model

        if m_score > pre_best:
            print("Save BEST model!")
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "checkpoint/", 'model_{}'.format("BEST")))  # Save model params
            torch.save((n_epi, pre_best, scores, m_scores),
                       os.path.join(save_dir, "checkpoint/", 'indicator_{}'.format("BEST")))  # Save data
            pre_best = m_score
        if n_epi % 100 == 0 and n_epi > 0:
            print("Save LATEST model!")
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "checkpoint", 'model_{}'.format("LATEST")))  # Save model params
            torch.save((n_epi, pre_best, scores, m_scores),
                       os.path.join(save_dir, "checkpoint", 'indicator_{}'.format("LATEST")))  # Save data

    env.close()


if __name__ == '__main__':
    main()

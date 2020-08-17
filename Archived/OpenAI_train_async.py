import warnings
import argparse
import collections
import time
import os
import gym
from gym_fightingice.envs.Machete import Machete
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from datetime import datetime
from OppModeling.atari_wrappers import make_ftg_display


Transition = collections.namedtuple('Transition', field_names=[
                                    'state', 'action', 'reward', 'done', 'new_state'])

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


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        output_conv = self.conv(x).view(x.size()[0], -1)
        output = self.fc(output_conv)
        return output



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        # in order to acclerate calculate
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])

        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play(self, net, epsilon=0.0, device=torch.device("cpu")):
        done_reward = None
        state = self.state
        self.env.render()
        if np.random.random() < epsilon:
            # take a random action
            action = self.env.action_space.sample()
        else:
            # get a max value aciton from the q-table
            state_vector = torch.tensor(
                np.array([state], copy=False)).to(device)
            qvals_vector = net(state_vector)
            _, act_v = torch.max(qvals_vector, dim=1)
            action = int(act_v.item())

        # get transition from the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        # add transitions into replay buffer for later sample
        trans = Transition(state, action, reward, is_done, new_state)
        self.replay_buffer.append(trans)

        # update state
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward



def calc_loss(batch, policy_net, target_net, gamma, is_double=False, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    # transfomr np arrary to tensor
    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # get values of actons in state_v
    state_action_values = policy_net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = target_net(next_states_v).max(1)[0]
        # done mask
        next_state_values[done_mask] = 0.0
        if is_double:
            next_state_acts = policy_net(next_states_v).max(1)[1]
            next_state_acts = next_state_acts.unsqueeze(-1)
            next_state_vals = target_net(next_states_v).gather(
                1, next_state_acts).squeeze(-1).detach()
        else:
            # not influence the net
            next_state_vals = next_state_values.detach()

    # calculate expect aciton values (Q-table)
    expected_state_action_values = next_state_vals * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default="FightingiceDisplayFrameskip-v0", action="store_true", help="Enable cuda")
    parser.add_argument("-d", "--double", default=False, action="store_true", help="enable double dqn")
    parser.add_argument("-n", "--name", default='Boxing', help="training name")
    parser.add_argument("--lr", default=0.0001, help="learning_rate")
    parser.add_argument("--batch_size", default=32, help="training batch_size")
    parser.add_argument("--seed", default=1234, help="training batch_size")
    parser.add_argument("--model", help="Model file to load")
    parser.add_argument("--stop_reward", default=400, help="Model file to load")
    parser.add_argument("--replay_size", default=100000, help="Model file to load")
    parser.add_argument("--replay_start", default=10000, help="Model file to load")
    parser.add_argument("--target_net_sync", default=1000, help="Model file to load")
    parser.add_argument("--epsilon_frames", default=1e6 ,help="Model file to load")
    parser.add_argument("--epsilon_start", default=1, help="Model file to load")
    parser.add_argument("--epsilon_final", default=0.01, help="Model file to load")
    parser.add_argument("--gamma", default=0.99, help="Model file to load")
    parser.add_argument("--save_dir", default='./OpenAI/DQN', help="Model file to load")
    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # mp.set_start_method("spawn")
    warnings.simplefilter("ignore", category=UserWarning)
    torch.manual_seed(args.seed)

    # init tensorboard
    save_dir = os.path.join(args.name, 'checkpoint')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tensorboard_dir = os.path.join(save_dir, "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    device = torch.device("cuda" if args.cuda else "cpu")

    # init policyNet and targetNet
    env = make_ftg_display(args.env, p2=Machete, port=4000)

    policy_net = DQN(env.observation_space.shape, env.action_space.n)
    target_net = DQN(env.observation_space.shape, env.action_space.n)
    # shared_policy_net = copy.deepcopy(policy_net)
    # shared_target_net = copy.deepcopy(target_net)
    policy_net.to(device)
    target_net.to(device)

    # init agent and replayBuffer

    memory = ReplayBuffer(args.replay_size)
    agent = Agent(env, memory)
    # training optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    gym.logger.set_level(gym.logger.INFO)  # Disable Gym warnings

    total_rewards = []
    frame = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    # load from model
    if args.model:
        assert torch.load(args.model)
        state = torch.load(args.model)
        policy_net.load_state_dict(state)
        args.epsilon_start = 0.01
        print('load model')
    # training loop
    while True:
        frame += 1
        epsilon = max(args.epsilon_final, args.epsilon_start -
                      frame / args.epsilon_frames)

        reward = agent.play(policy_net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame - ts_frame) / (time.time() - ts)
            ts_frame = frame
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("frame-{frame}: finish {game_num} games, reward {reward:.3f}, speed {speed:.2f} f/s".format(
                frame=frame, game_num=len(total_rewards), reward=m_reward, speed=speed
            ))

            # update tensorboard
            writer.add_scalar("reward/iteration", reward, frame)
            writer.add_scalar("reward/avg_100", m_reward, frame)
            writer.add_scalar("indicator/speed", speed, frame)
            writer.add_scalar("indicator/epsilon", epsilon, frame)

            # save best model every 100 frame and chech whether the training is done
            # each game have 1784 frame now
            every_save_epoch = len(total_rewards) % 2
            if every_save_epoch is 0:
                if best_m_reward is None:
                    best_m_reward = m_reward
                elif best_m_reward < m_reward and m_reward > 0:
                    torch.save(policy_net.state_dict(),
                               "{env}-{name}-best_{reward:.0f}.dat".format(env=args.env, name=args.name,
                                                                           reward=m_reward))
                    best_m_reward = m_reward

                if m_reward > args.stop_reward:
                    print("Solved in %d frames!" % frame)
                    break

        # procduce the first batches of transition from scratch
        # apply by agent.play
        if len(memory) < args.replay_start:
            continue

        # sync target_net
        if frame % args.target_net_sync == 0:
            target_net.load_state_dict(policy_net.state_dict())

        optimizer.zero_grad()

        # get a sample batch
        batch = memory.sample(args.batch_size)
        loss = calc_loss(batch, policy_net, target_net,
                         gamma=args.gamma, is_double=args.double, device=device)
        writer.add_scalar("loss/batch", loss / args.batch_size, frame)
        loss.backward()
        optimizer.step()
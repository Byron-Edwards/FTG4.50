import warnings
import time
import logging
import gym
import gym_fightingice
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import OpenAI.ByronAI.const as const
import OpenAI.ByronAI.args as args
import OpenAI.ByronAI.dqn_model as model
import collections
from torch.multiprocessing import Queue, Process, Manager
import multiprocessing as mp
logger = logging.getLogger("py4j")
logger.setLevel(logging.WARNING)
logger.addHandler(logging.StreamHandler())


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
    # get rid of warnings to have a cleaner terminal logger
    warnings.simplefilter("ignore", category=UserWarning)

    # set args and params
    params = const.HYPERPARAMS['optimize']
    args = args.get_arg(params)
    torch.manual_seed(args.seed)

    # init tensorboard
    writer = SummaryWriter(comment="-" + params.env + "-" + params.name)

    device = torch.device("cuda" if args.cuda else "cpu")

    # wappers of env
    env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path=".", port=4000)

    # init policyNet and targetNet
    policy_net = model.DQN(env.observation_space.shape,
                           env.action_space.n).to(device).share_memory()
    target_net = model.DQN(env.observation_space.shape,
                           env.action_space.n).to(device).share_memory()
    env.close()

    # init agent and replayBuffer
    buffer = model.ReplayBuffer(params.replay_size)
    # agents = [model.Agent(env, None, params.name) for env in envs]

    # training optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)

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
        params.epsilon_start = 0.01
        logging.info('load model')

    # multiprocessing set up
    # mp.set_start_method("spawn")
    q = Queue(maxsize=params.replay_start)

    def subprocess(portal,p2_name,net,queue,device):
        env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path=".", port=portal)
        agent = model.Agent(env, None, p2_name)
        agent.play_async(net, queue, device=device)

    processes = [Process(target=subprocess, args=(4000+i, params.name, policy_net, q, device)) for i in range(0, 10, 2)]
    [process.start() for process in processes]

    # training loop
    logging.info("training loop start")
    while True:
        frame += 1
        print("Frame: {}".format(frame))
        message = q.get()
        logging.debug("get {} {}".format(message[0],message[1]))
        if message[0] == 'done_reward':
            pid = message[1]
            logging.debug("receive done_reward from pid{}".format(pid))
            reward = message[2]
            total_rewards.append(reward)
            speed = (frame - ts_frame) / (time.time() - ts)
            ts_frame = frame
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])

            print("frame-{frame}: finish {game_num} games, latest_reward {reward:.3f}, mean_reward {m_reward:.3f}, speed {speed:.2f} f/s".format(
                frame=frame, game_num=len(total_rewards),reward=reward, m_reward=m_reward, speed=speed
            ))

            # update tensorboard
            writer.add_scalar("reward/iteration", reward, frame)
            writer.add_scalar("reward/avg_100", m_reward, frame)
            writer.add_scalar("indicator/speed", speed, frame)
            # writer.add_scalar("indicator/epsilon", epsilon, frame)

            # save best model every 100 frame and chech whether the training is done
            # each game have 1784 frame now
            every_save_epoch = len(total_rewards) % 2
            if every_save_epoch is 0:
                if best_m_reward is None:
                    best_m_reward = m_reward
                elif best_m_reward < m_reward and m_reward > -200:
                    torch.save(policy_net.state_dict(),
                               "./OpenAI/ByronAI/{env}-{name}-best_{reward:.0f}.dat".format(env=params.env, name=args.name,
                                                                           reward=m_reward))
                    best_m_reward = m_reward

                if m_reward > params.stop_reward:
                    print("Solved in %d frames!" % frame)
                    [p.close() for p in processes]
                    break

        # procduce the first batches of transition from scratch
        # apply by agent.play
        if message[0] == 'trans':
            pid = message[1]
            logging.debug("receive Trans from pid {}".format(pid))
            buffer.append(message[2])
            if len(buffer) < params.replay_start:
                continue
        writer.add_scalar("buffer/reward_mean", buffer.rewards_mean, frame)
        writer.add_scalar("buffer/reward_std", buffer.rewards_std, frame)
        # sync target_net

        if frame % params.target_net_sync == 0:
            target_net.load_state_dict(policy_net.state_dict())
            # [p.close() for p in processes]
            # processes = [Process(target=subprocess, args=(4000+i, params.name, policy_net, q, device)) for i in range(0, 10, 2)]
            # [process.start() for process in processes]

        optimizer.zero_grad()
        # get a sample batch
        batch = buffer.sample(args.batch_size)
        loss = calc_loss(batch, policy_net, target_net,
                         gamma=params.gamma, is_double=args.double, device=device)

        writer.add_scalar("loss/batch", loss / args.batch_size, frame)
        loss.backward()
        optimizer.step()

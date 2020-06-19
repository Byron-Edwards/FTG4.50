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
logger = logging.getLogger("py4j")
logger.setLevel(logging.WARNING)


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
    env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path=".", port=4001)


    # init policyNet and targetNet
    policy_net = model.DQN(env.observation_space.shape,
                           env.action_space.n).to(device)
    target_net = model.DQN(env.observation_space.shape,
                           env.action_space.n).to(device)

    # init agent and replayBuffer
    buffer = model.ReplayBuffer(params.replay_size)
    agent = model.Agent(env, buffer, params.name)

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
        if args.data:
            buffer, epsilon, best_m_reward, frame = torch.load(args.data)
            agent.replay_buffer = buffer
            agent.replay_buffer.pre_transition = None
        logging.info('load model from {}'.format(args.model))
    # training loop
    logging.info("training loop start")
    while True:
        frame += 1
        logging.debug("Frame: {}".format(frame))
        epsilon = max(params.epsilon_final, params.epsilon_start -
                      frame / params.epsilon_frames)
        # epsilon = 0.1
        reward = agent.play(policy_net, epsilon, device=device)


        if reward is not None:
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
            writer.add_scalar("indicator/epsilon", epsilon, frame)

            # save best model every 100 frame and chech whether the training is done
            # each game have 1784 frame now
            # every_save_epoch = len(total_rewards) % 2
            # if every_save_epoch is 0:

            if best_m_reward is None:
                best_m_reward = m_reward
            elif best_m_reward < m_reward and (m_reward > -250) and len(buffer) >= params.replay_start:

                torch.save(policy_net.state_dict(),
                           "./OpenAI/ByronAI/{env}-{name}-best_{reward:.0f}.dat".format(env=params.env, name=args.name,
                                                                       reward=m_reward))
                best_m_reward = m_reward
                all_data = (buffer, epsilon,total_rewards, best_m_reward, frame,)
                torch.save(all_data,
                           "./OpenAI/ByronAI/{env}-{name}-best_{reward:.0f}_data".format(env=params.env, name=args.name,
                                                                       reward=m_reward))
            if m_reward > params.stop_reward:
                print("Solved in %d frames!" % frame)
                break

        # procduce the first batches of transition from scratch
        # apply by agent.play

        if len(buffer) < params.replay_start:
            continue
        writer.add_scalar("buffer/reward_mean", buffer.rewards_mean, frame)
        writer.add_scalar("buffer/reward_std", buffer.rewards_std, frame)
        # sync target_net

        if frame % params.target_net_sync == 0:
            target_net.load_state_dict(policy_net.state_dict())

        optimizer.zero_grad()
        # get a sample batch
        batch = buffer.sample(args.batch_size)
        loss = calc_loss(batch, policy_net, target_net,
                         gamma=params.gamma, is_double=args.double, device=device)

        writer.add_scalar("loss/batch", loss / args.batch_size, frame)
        loss.backward()
        optimizer.step()

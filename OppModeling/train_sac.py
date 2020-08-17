import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from OppModeling.SAC import MLPActorCritic
from OppModeling.ReplayBuffer import ReplayBuffer, ReplayBufferOppo, ReplayBufferShare
from OppModeling.CPC import CDCK2
from OppModeling.atari_wrappers import make_ftg_ram,make_ftg_ram_nonstation
from OppModeling.model_parameter_trans import state_dict_trans


def sac(global_ac, global_ac_targ, rank, T, E, args, scores, wins,buffer, device=None, tensorboard_dir=None,):
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
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
    local_ac = MLPActorCritic(obs_dim, act_dim, **dict(hidden_sizes=[args.hid] * args.l)).to(device)
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
    alpha = max(local_ac.log_alpha.exp().item(), args.min_alpha) if not args.fix_alpha else args.min_alpha

    # Set up optimizers for policy and q-function
    # Async Version
    pi_optimizer = Adam(global_ac.pi.parameters(), lr=args.lr, eps=1e-4)
    q1_optimizer = Adam(global_ac.q1.parameters(), lr=args.lr, eps=1e-4)
    q2_optimizer = Adam(global_ac.q2.parameters(), lr=args.lr, eps=1e-4)
    alpha_optim = Adam([global_ac.log_alpha], lr=args.lr, eps=1e-4)

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0
    trajectory = list()
    discard = False
    t = T.value()
    e = E.value()
    # Main loop: collect experience in env and update/log each epoch
    while e <= args.total_episode:

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > args.start_steps:
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
        d = False if (ep_len == args.max_ep_len) or discard else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        T.increment()
        t = T.value()

        # End of trajectory handling
        if d or (ep_len == args.max_ep_len) or discard:
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
                    rank, args.p2, e, t, ep_ret, m_score, win_rate, ep_len, alpha))
            writer.add_scalar("metrics/round_score", ep_ret, e)
            writer.add_scalar("metrics/mean_score", m_score.item(), e)
            writer.add_scalar("metrics/win_rate", win_rate.item(), e)
            writer.add_scalar("metrics/round_step", ep_len, e)
            writer.add_scalar("metrics/alpha", alpha, e)
            o, ep_ret, ep_len = env.reset(), 0, 0
            trajectory = list()
            discard = False

        # Update handling
        if t >= args.update_after and t % args.update_every == 0:
            for j in range(args.update_every):

                batch = replay_buffer.sample_batch(args.batch_size,device=device)
                # First run one gradient descent step for Q1 and Q2
                q1_optimizer.zero_grad()
                q2_optimizer.zero_grad()
                loss_q = local_ac.compute_loss_q(batch, global_ac_targ, args.gamma, alpha)
                loss_q.backward()

                # Next run one gradient descent step for pi.
                pi_optimizer.zero_grad()
                loss_pi, entropy = local_ac.compute_loss_pi(batch, alpha)
                loss_pi.backward()

                alpha_optim.zero_grad()
                alpha_loss = -(local_ac.log_alpha * (entropy + target_entropy).detach()).mean()
                alpha_loss.backward(retain_graph=False)
                alpha = max(local_ac.log_alpha.exp().item(), args.min_alpha) if not args.fix_alpha else args.min_alpha

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
                        p_targ.data.copy_((1 - args.polyak) * p.data + args.polyak * p_targ.data)

                writer.add_scalar("training/pi_loss", loss_pi.detach().item(), t)
                writer.add_scalar("training/q_loss", loss_q.detach().item(), t)
                writer.add_scalar("training/alpha_loss", alpha_loss.detach().item(), t)
                writer.add_scalar("training/entropy", entropy.detach().mean().item(), t)

        if t % args.save_freq == 0 and t > 0:
            torch.save(global_ac.state_dict(), os.path.join(args.save_dir, args.exp_name, args.model_para))
            state_dict_trans(global_ac.state_dict(), os.path.join(args.save_dir, args.exp_name,  args.numpy_para))
            torch.save((e, t, list(scores), list(wins)), os.path.join(args.save_dir, args.exp_name, args.train_indicator))
            print("Saving model at episode:{}".format(t))


def sac_opp(global_ac, global_ac_targ, rank, T, E, args, scores, wins, buffer, device=None, tensorboard_dir=None, ):
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
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
    local_ac = MLPActorCritic(obs_dim, act_dim, **dict(hidden_sizes=[args.hid] * args.l)).to(device)
    local_ac.load_state_dict(global_ac.state_dict())
    local_cpc = CDCK2(timestep, obs_dim, z_dim, c_dim)
    local_cpc.load_state_dict(global_cpc.state_dict())
    print("local ac load global ac")

    for p in global_ac_targ.parameters():
        p.requires_grad = False

    replay_buffer = ReplayBufferOppo(max_size=args.replay_size,encoder=local_cpc)

    # Entropy Tuning
    target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()  # heuristic value from the paper
    alpha = max(local_ac.log_alpha.exp().item(), args.min_alpha) if not args.fix_alpha else args.min_alpha

    pi_optimizer = Adam(global_ac.pi.parameters(), lr=args.lr, eps=1e-4)
    q1_optimizer = Adam(global_ac.q1.parameters(), lr=args.lr, eps=1e-4)
    q2_optimizer = Adam(global_ac.q2.parameters(), lr=args.lr, eps=1e-4)
    alpha_optim = Adam([global_ac.log_alpha], lr=args.lr, eps=1e-4)

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0
    trajectory = list()
    discard = False
    t = T.value()
    e = E.value()
    while e <= args.total_episode:
        if t > args.start_steps:
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
        d = False if (ep_len == args.max_ep_len) or discard else d

        trajectory.append(dict(obs=o, action=a, reward=r, next_obs=o2, done=d,latent=None))

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        T.increment()
        t = T.value()
        # End of trajectory handling
        if d or (ep_len == args.max_ep_len) or discard:
            replay_buffer.store(trajectory)
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
                    rank, args.p2, e, t, ep_ret, m_score, win_rate, ep_len, alpha))
            writer.add_scalar("metrics/round_score", ep_ret, e)
            writer.add_scalar("metrics/mean_score", m_score.item(), e)
            writer.add_scalar("metrics/win_rate", win_rate.item(), e)
            writer.add_scalar("metrics/round_step", ep_len, e)
            writer.add_scalar("metrics/alpha", alpha, e)
            o, ep_ret, ep_len = env.reset(), 0, 0
            trajectory = list()
            discard = False

        # CPC update handing


        # SAC Update handling
        if t >= args.update_after and t % args.update_every == 0:
            for j in range(args.update_every):

                batch = replay_buffer.sample_trans(batch_size=args.batch_size,device=device)
                # First run one gradient descent step for Q1 and Q2
                q1_optimizer.zero_grad()
                q2_optimizer.zero_grad()
                loss_q = local_ac.compute_loss_q(batch, global_ac_targ, args.gamma, alpha)
                loss_q.backward()

                # Next run one gradient descent step for pi.
                pi_optimizer.zero_grad()
                loss_pi, entropy = local_ac.compute_loss_pi(batch, alpha)
                loss_pi.backward()

                alpha_optim.zero_grad()
                alpha_loss = -(local_ac.log_alpha * (entropy + target_entropy).detach()).mean()
                alpha_loss.backward(retain_graph=False)
                alpha = max(local_ac.log_alpha.exp().item(), args.min_alpha) if not args.fix_alpha else args.min_alpha

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
                        p_targ.data.copy_((1 - args.polyak) * p.data + args.polyak * p_targ.data)

                writer.add_scalar("training/pi_loss", loss_pi.detach().item(), t)
                writer.add_scalar("training/q_loss", loss_q.detach().item(), t)
                writer.add_scalar("training/alpha_loss", alpha_loss.detach().item(), t)
                writer.add_scalar("training/entropy", entropy.detach().mean().item(), t)

        if t % args.save_freq == 0 and t > 0:
            torch.save(global_ac.state_dict(), os.path.join(args.save_dir, args.exp_name, args.model_para))
            state_dict_trans(global_ac.state_dict(), os.path.join(args.save_dir, args.exp_name,  args.numpy_para))
            torch.save((e, t, list(scores), list(wins)), os.path.join(args.save_dir, args.exp_name, args.train_indicator))
            print("Saving model at episode:{}".format(t))
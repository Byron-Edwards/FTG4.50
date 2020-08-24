import os
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from OppModeling.atari_wrappers import make_ftg_ram,make_ftg_ram_nonstation


def sac(global_ac, rank, E, args, buffer_q, device=None, tensorboard_dir=None,):
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    # writer = GlobalSummaryWriter.getSummaryWriter()
    tensorboard_dir = os.path.join(tensorboard_dir, str(rank))
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    if args.exp_name == "test":
        env = gym.make("CartPole-v0")
    elif args.non_station:
        env = make_ftg_ram_nonstation(args.env, p2_list=args.list, total_episode=args.station_rounds,stable=args.stable)
    else:
        env = make_ftg_ram(args.env, p2=args.p2)
    print("set up child process env")

    # Prepare for interaction with environment
    scores, wins = [], []
    o, ep_ret, ep_len = env.reset(), 0, 0
    discard = False
    local_t, local_e = 0, 0
    # Main loop: collect experience in env and update/log each epoch
    while E.value() <= args.episode:
        with torch.no_grad():
            a = global_ac.get_action(o, device=device)
        # Step the env
        o2, r, d, info = env.step(a)
        if info.get('no_data_receive', False):
            discard = True
        ep_ret += r
        ep_len += 1

        d = False if (ep_len == args.max_ep_len) or discard else d
        # send the transition to main process
        if hasattr(env, 'p2'):
            p2 = env.p2
        else:
            p2 = None
        transition = (o, a, r, o2, d, str(p2))
        buffer_q.put(transition,)
        o = o2
        local_t += 1
        # End of trajectory handling
        if d or (ep_len == args.max_ep_len) or discard:
            E.increment()
            e = E.value()
            local_e += 1
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            if info.get('win', False):
                wins.append(1)
            else:
                wins.append(0)
            scores.append(ep_ret)
            m_score = np.mean(scores[-100:])
            win_rate = np.mean(wins[-100:])
            print(
                "Process {}, opponent:{}, # of global_episode :{}, round score: {}, mean score : {:.1f}, win_rate:{}, steps: {}".format(
                    rank, args.p2, e, ep_ret, m_score, win_rate, ep_len))
            writer.add_scalar("actor/round_score", ep_ret, e)
            writer.add_scalar("actor/mean_score", m_score.item(), e)
            writer.add_scalar("actor/win_rate", win_rate.item(), e)
            writer.add_scalar("actor/round_step", ep_len, e)
            o, ep_ret, ep_len = env.reset(), 0, 0
            discard = False


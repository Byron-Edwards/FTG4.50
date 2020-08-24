import os
import gym
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from OppModeling.atari_wrappers import make_ftg_ram, make_ftg_ram_nonstation


def test_proc(global_ac, env, args, device):
    scores, wins, m_score, win_rate = [], [], 0, 0
    o, ep_ret, ep_len = env.reset(), 0, 0
    discard = False
    local_t = 0
    local_e = 0
    while local_e < args.test_episode:
        with torch.no_grad():
            a = global_ac.get_action(o, greedy=True, device=device)
        # Step the env
        o2, r, d, info = env.step(a)
        if info.get('no_data_receive', False):
            discard = True
        ep_ret += r
        ep_len += 1
        d = False if (ep_len == args.max_ep_len) or discard else d
        o = o2
        local_t += 1
        # End of trajectory handling
        if d or (ep_len == args.max_ep_len) or discard:
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            local_e += 1
            if info.get('win', False):
                wins.append(1)
            else:
                wins.append(0)
            scores.append(ep_ret)
            o, ep_ret, ep_len = env.reset(), 0, 0
            discard = False
    m_score = np.mean(scores)
    win_rate = np.mean(wins)
    return m_score, win_rate, local_t


def test_summary(p2, steps, m_score, win_rate, writer, args, e):
    print("\n" + "=" * 20 + "TEST SUMMARY" + "=" * 20)
    summary = "opponent:\t{}\n# test episode:\t{}\n# total steps:\t{}\nmean score:\t{:.1f}\nwin_rate:\t{}".format(
        p2, args.test_episode, steps, m_score, win_rate)
    print(summary)
    print("=" * 20 + "TEST SUMMARY" + "=" * 20 + "\n")
    writer.add_scalar("Test/mean_score", m_score.item(), e)
    writer.add_scalar("Test/win_rate", win_rate.item(), e)
    writer.add_scalar("Test/total_step", steps, e)


def test_func(global_ac, rank, E, args, device, tensorboard_dir, ):
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    print("set up Test process env")
    non_station_dir = os.path.join(tensorboard_dir, "test_{}".format("non_station"))
    if not os.path.exists(non_station_dir):
        os.makedirs(non_station_dir)
    non_station_writer = SummaryWriter(log_dir=non_station_dir)
    writers = []
    for p2 in args.list:
        temp_dir = os.path.join(tensorboard_dir, "test_{}".format(p2))
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        writer = SummaryWriter(log_dir=temp_dir)
        writers.append(writer)
    last_test = 0
    # Main loop: collect experience in env and update/log each epoch
    while E.value() <= args.episode:
        e = E.value()
        if e == last_test:
            continue
        last_test = e
        if e > 0 and e % args.test_every==0:
            # non_station evaluation
            if args.exp_name == "test":
                env = gym.make("CartPole-v0")
            else:
                env = make_ftg_ram_nonstation(args.env, p2_list=args.list, total_episode=args.test_episode,
                                              stable=args.stable)
            m_score, win_rate, steps = test_proc(global_ac, env, args, device)
            test_summary("Non-Station", steps, m_score, win_rate, non_station_writer, args, e)
            env.close()
            # station evaluation
            for index, p2 in enumerate(args.list):
                if args.exp_name == "test":
                    env = gym.make("CartPole-v0")
                else:
                    env = make_ftg_ram(args.env, p2=p2)
                m_score, win_rate, steps = test_proc(global_ac, env, args, device)
                test_summary(p2, steps, m_score, win_rate, writers[index], args, e)
                env.close()


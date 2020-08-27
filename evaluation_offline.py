import os
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from copy import  deepcopy
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from OppModeling.atari_wrappers import make_ftg_ram, make_ftg_ram_nonstation
from OppModeling.SAC import MLPActorCritic

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


def test_func(global_ac, rank, e, p2, args, device, tensorboard_dir, ):
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    print("set up Test process env")
    temp_dir = os.path.join(tensorboard_dir, "test_{}".format(p2))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    writer = SummaryWriter(log_dir=temp_dir)

    # Main loop: collect experience in env and update/log each epoch
    if args.exp_name == "test":
        env = gym.make("CartPole-v0")
    elif p2 == "Non-station":
        env = make_ftg_ram_nonstation(args.env, p2_list=args.list, total_episode=args.test_episode,stable=args.stable)
    else:
        env = make_ftg_ram(args.env, p2=p2)
    print("TESTING process {} start to test, opp: {}".format(rank, p2))
    m_score, win_rate, steps = test_proc(global_ac, env, args, device)
    test_summary(p2, steps, m_score, win_rate, writer, args, e)
    env.close()
    del env
    print("TESTING process {} finished, opp: {}".format(rank, p2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # running setting
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--n_process', type=int, default=1)
    # basic env setting
    parser.add_argument('--env', type=str, default="FightingiceDataFrameskip-v0")
    parser.add_argument('--p2', type=str, default="Toothless")
    # non station agent settings
    parser.add_argument('--non_station', default=False, action='store_true')
    parser.add_argument('--stable', default=False, action='store_true')
    parser.add_argument('--station_rounds', type=int, default=1000)
    parser.add_argument('--list', nargs='+')
    # training setting
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2, help="layers")
    parser.add_argument('--test_episode', type=int, default=10)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    # CPC setting
    parser.add_argument('--cpc', default=False, action="store_true")
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--c_dim', type=int, default=32)
    parser.add_argument('--timestep', type=int, default=10)
    # Saving settings
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--save-dir', type=str, default="./experiments")
    parser.add_argument('--traj_dir', type=str, default="./experiments")
    parser.add_argument('--model_para', type=str, default="test_sac.torch")
    parser.add_argument('--cpc_para', type=str, default="test_cpc.torch")
    parser.add_argument('--numpy_para', type=str, default="test.numpy")
    parser.add_argument('--train_indicator', type=str, default="test.data")
    args = parser.parse_args()

    mp.set_start_method("forkserver")
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    experiment_dir = os.path.join(args.save_dir, args.exp_name)
    tensorboard_dir = os.path.join(experiment_dir, "evaluation")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    file_list = os.listdir(experiment_dir)
    print()
    model_para = [i for i in file_list if "model_torch" in i]
    model_para = ["model_torch_{}".format(i*100) for i in range(len(model_para)) if "model_torch_{}".format(i*100) in model_para ]
    obs_dim = 143
    act_dim = 56
    ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)
    p2_list = ["Non-station"] + args.list
    global_ac = MLPActorCritic(obs_dim, act_dim, **ac_kwargs)

    scores, win_rates,rounds = [[], [], [], [],], [[], [], [], []], [[], [], [], []]
    for e in range(2):
        global_ac.load_state_dict(torch.load(os.path.join(experiment_dir, model_para[e])))
        global_ac.share_memory()
        for index, p2 in enumerate(p2_list):
            if args.exp_name == "test":
                env = gym.make("CartPole-v0")
            elif p2 == "Non-station":
                env = make_ftg_ram_nonstation(args.env, p2_list=args.list, total_episode=args.test_episode,
                                              stable=args.stable)
            else:
                env = make_ftg_ram(args.env, p2=p2)
            m_score, win_rate, steps = test_proc(global_ac, env, args, torch.device("cpu"))
            scores[index].append(m_score)
            win_rates[index].append(win_rate)
            rounds[index].append(e*100)
            env.close()
            del env
        print("First Round finished")
        torch.save((scores, win_rates, rounds,), os.path.join(tensorboard_dir, "test_data_{}".format(e*100)))

    for index in range(len(scores)):
        plt.plot(rounds[index], scores[index], label=p2_list[index])
    plt.legend(title='Opponent')
    plt.savefig(os.path.join(tensorboard_dir, "evaluation_plot"))



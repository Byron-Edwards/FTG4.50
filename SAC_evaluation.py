import gym
import torch
import os
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from OppModeling.SAC import MLPActorCritic
from OppModeling.CPC import CPC
from OppModeling.ReplayBuffer import ReplayBuffer
from OppModeling.atari_wrappers import make_ftg_ram_nonstation, make_ftg_ram


def ood_scores(prob):
    assert prob.ndim == 2
    data = torch.tensor(prob, dtype=torch.float32)
    max_softmax, _ = torch.max(data, dim=1)
    uncertainty = 1 - max_softmax
    return uncertainty


def get_ood_hist(replay_buffer, batch_size):
    batch = replay_buffer.sample_batch(batch_size, device=device)
    o, a, r, o2, d = batch['obs'], batch['act'], batch['rew'], batch['obs2'], batch['done']
    with torch.no_grad():
        a_prob, log_a_prob, sample_a, max_a, = global_ac.get_actions_info(global_ac.pi(o))
    uncertainty = ood_scores(a_prob)
    return uncertainty


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # running setting
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--episode', type=int, default=10)
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
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--replay_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--l', type=int, default=2, help="layers")
    # CPC setting
    parser.add_argument('--cpc', default=False, action="store_true")
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--c_dim', type=int, default=32)
    parser.add_argument('--timestep', type=int, default=10)
    # Saving settings
    parser.add_argument('--exp_name', type=str, default='ReiwaThunder')
    parser.add_argument('--save-dir', type=str, default="./experiments")
    parser.add_argument('--traj_dir', type=str, default="./experiments")
    parser.add_argument('--model_para', type=str, default="ReiwaThunder_1.torch")
    parser.add_argument('--cpc_para', type=str, default="ReiwaThunder.torch")
    args = parser.parse_args()

    experiment_dir = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    test_save_dir = os.path.join(experiment_dir, "evaluation")
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)

    tensorboard_dir = os.path.join(test_save_dir, args.p2)
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)

    writer = SummaryWriter(log_dir=tensorboard_dir)

    ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)
    device = torch.device("cuda") if args.cuda else torch.device("cpu")

    if args.exp_name == "test":
        env = gym.make("CartPole-v0")
    elif args.non_station:
        env = make_ftg_ram_nonstation(args.env, p2_list=args.list, total_episode=args.station_rounds,stable=args.stable)
    else:
        env = make_ftg_ram(args.env, p2=args.p2)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    if args.cpc:
        global_ac = MLPActorCritic(obs_dim + args.c_dim, act_dim, **ac_kwargs)
        global_cpc = CPC(timestep=args.timestep, obs_dim=obs_dim, hidden_sizes=[args.hid] * args.l, z_dim=args.z_dim,c_dim=args.c_dim)
    else:
        global_ac = MLPActorCritic(obs_dim, act_dim, **ac_kwargs)
        global_cpc = None

    if os.path.exists(os.path.join(args.save_dir, args.exp_name, args.model_para)):
        global_ac.load_state_dict(torch.load(os.path.join(args.save_dir, args.exp_name, args.model_para)))
        print("load sac model")

    if os.path.exists(args.cpc_para) and args.cpc:
        global_cpc.load_state_dict(torch.load(args.cpc_para))
        print("load cpc model")

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=args.replay_size)
    o, ep_ret, ep_len = env.reset(), 0, 0
    if args.cpc:
        c_hidden = global_cpc.init_hidden(1, args.c_dim, use_gpu=args.cuda)
        c1, c_hidden = global_cpc.predict(o, c_hidden)
        assert len(c1.shape) == 3
        c1 = c1.flatten().cpu().numpy()
        round_embedding = []
        all_embeddings = []
        meta = []
    trajectory = list()
    p2 = env.p2
    discard = False
    wins, scores, win_rate, m_score = [], [], 0, 0
    local_t, local_e = 0, 0
    while local_e < args.episode:
        with torch.no_grad():
            a = global_ac.get_action(o, greedy=True, device=device)

        # Step the env
        o2, r, d, info = env.step(a)
        if info.get('no_data_receive', False):
            discard = True
        ep_ret += r
        ep_len += 1

        d = False if (ep_len == args.max_ep_len) or discard else d

        replay_buffer.store(o, a, r, o2, d)

        if args.cpc:
            # changed the trace structure for further analysis
            c2, c_hidden = global_cpc.predict(o2, c_hidden)
            assert len(c2.shape) == 3
            c2 = c2.flatten().cpu().numpy()
            trajectory.append([o, a, r, o2, d, c1, c2, ep_len])
            round_embedding.append(c1)
            all_embeddings.append(c1)
            meta.append([env.p2,local_e, ep_len,r, a])
            c1 = c2

        o = o2
        local_t += 1

        # End of trajectory handling
        if d or (ep_len == args.max_ep_len) or discard:
            local_e += 1
            if info.get('win', False):
                wins.append(1)
            else:
                wins.append(0)
            scores.append(ep_ret)
            m_score = np.mean(scores[-100:])
            win_rate = np.mean(wins[-100:])
            print(
                "opponent:{}, # of episode :{},  # of steps :{}, round score: {}, mean score : {:.1f}, win_rate:{}, steps: {}".format(
                    p2, local_e, local_t, ep_ret, m_score, win_rate, ep_len))
            writer.add_scalar("metrics/round_score", ep_ret, local_e)
            writer.add_scalar("metrics/mean_score", m_score.item(), local_e)
            writer.add_scalar("metrics/win_rate", win_rate.item(), local_e)
            writer.add_scalar("metrics/round_step", ep_len, local_e)

            # write data for the ood calculation
            uncertainty = get_ood_hist(replay_buffer, args.batch_size)
            writer.add_histogram(values=uncertainty, max_bins=100, global_step=local_e, tag="opp")
            if args.cpc:
                round_embedding = np.array(round_embedding)
                writer.add_embedding(mat=all_embeddings, metadata=meta,metadata_header=["opponent", "round", "step", "reward", "action"])
                c_hidden = global_cpc.init_hidden(1, args.c_dim, use_gpu=args.cuda)
                round_embedding = []
                trajectory = list()

            o, ep_ret, ep_len = env.reset(), 0, 0
            discard = False

            # if p2 changed, save the summary and reset the indicators
            if env.p2 != p2:
                print("=" * 20 + "TEST SUMMARY" + "=" * 20)
                summary = "opponent:\t{}\n# of episode:\t{}\n# of steps:\t{}\nmean score:\t{:.1f}\nwin_rate:\t{}\n".format(
                    p2, local_e, local_t, m_score, win_rate)
                print(summary)
                print("=" * 20 + "TEST SUMMARY" + "=" * 20)
                with open(os.path.join(test_save_dir, p2 + "_summary.txt"), 'w') as f:
                    f.write(summary)

                replay_buffer.reset()
                wins, scores, win_rate, m_score = [], [], 0, 0
                local_t, local_e = 0, 0
            p2 = env.p2

    # Test end summary and saving
    env.close()
    del env





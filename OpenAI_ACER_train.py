# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import copy
import cv2
# import platform
import gym
import gym_fightingice
import torch
from torch import multiprocessing as mp
from time import sleep
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter



from OpenAI.ACER.model import ActorCritic
from OpenAI.ACER.optim import SharedRMSprop
from OpenAI.ACER.memory import EpisodicReplayMemory
from OpenAI.ACER.utils import Counter
from ACER_train import off_policy_train
from ACER_train import actor


parser = argparse.ArgumentParser(description='ACER')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--cuda', type=bool, default=True, help='cuda Device')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='Number of training async acotrs (does not include single validation agent)')
parser.add_argument('--T-max', type=int, default=100000000, metavar='STEPS', help='Number of training steps')
parser.add_argument('--t-max', type=int, default=500, metavar='STEPS',
                    help='Max number of forward steps for A3C before update')
parser.add_argument('--max-episode-length', type=int, default=500, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--hidden-size', type=int, default=128, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--model', default="./OpenAI/OpenAI_ACER/checkpoint/model_LATEST", type=str, metavar='PARAMS',
                    help='Pretrained model (state dict)')
parser.add_argument('--memory', default="./OpenAI/OpenAI_ACER/checkpoint/memory", type=str, metavar='PARAMS',
                    help='Pretrained memory (state dict)')
parser.add_argument('--data', default="./OpenAI/OpenAI_ACER/checkpoint/indicator_LATEST", type=str, metavar='PARAMS',
                    help='Pretrained data (state dict)')
parser.add_argument('--on-policy', action='store_true', help='Use pure on-policy training (A3C)')
parser.add_argument('--memory-capacity', type=int, default=100000, metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-ratio', type=int, default=4, metavar='r', help='Ratio of off-policy to on-policy updates')
parser.add_argument('--replay-start', type=int, default=1000, metavar='EPISODES',
                    help='Number of transitions to save before starting off-policy training')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--trace-decay', type=float, default=1, metavar='λ', help='Eligibility trace decay factor')
parser.add_argument('--trace-max', type=float, default=10, metavar='c', help='Importance weight truncation (max) value')
parser.add_argument('--trust-region', default=True, action='store_true', help='Use trust region')
parser.add_argument('--trust-region-decay', type=float, default=0.99, metavar='α',
                    help='Average model weight decay rate')
parser.add_argument('--trust-region-threshold', type=float, default=1, metavar='δ', help='Trust region threshold value')
parser.add_argument('--reward-clip', action='store_true', help='Clip rewards to [-1, 1]')
parser.add_argument('--lr', type=float, default=1e-4, metavar='η', help='Learning rate')
parser.add_argument('--lr-decay', default=False, action='store_true', help='Linearly decay learning rate to 0')
parser.add_argument('--lr-min', default=1e-6, type=float, help='minimal learning rate')
parser.add_argument('--rmsprop-decay', type=float, default=0.99, metavar='α', help='RMSprop decay factor')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Off-policy batch size')
parser.add_argument('--entropy-weight', type=float, default=0.001, metavar='β', help='Entropy regularisation weight')
parser.add_argument('--max-gradient-norm', type=float, default=40, metavar='VALUE', help='Gradient L2 normalisation')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=25000, metavar='STEPS',
                    help='Number of training steps between evaluations (roughly)')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N',
                    help='Number of evaluation episodes to average over')
parser.add_argument('--render', action='store_true', help='Render evaluation agent')
parser.add_argument('--name', type=str, default='./OpenAI/OpenAI_ACER', help='Save folder')
parser.add_argument('--env', type=str, default='FightingiceDataNoFrameskip-v0', help='environment name')
parser.add_argument('--port', type=int, default=4000, help='FightingICE running Port')
parser.add_argument('--p2', type=str, default="RHEA_PI", help='FightingICE running Port')

if __name__ == '__main__':
    # BLAS setup
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    mp.set_start_method('forkserver')

    # Setup
    args = parser.parse_args()

    # To solve the open file issue
    mp.set_sharing_strategy('file_system')
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (args.nofile, rlimit[1]))
    # print("Set open file number to {}".format(args.nofile))

    torch.manual_seed(args.seed)
    gym.logger.set_level(gym.logger.ERROR)  # Disable Gym warnings
    device = torch.device('cuda' if args.cuda else "cpu")

    # Creating directories.
    save_dir = os.path.join(args.name, 'checkpoint')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tensorboard_dir = os.path.join(save_dir, "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    print(' ' * 26 + 'Options')

    # Saving parameters
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for k, v in vars(args).items():
            print(' ' * 26 + k + ': ' + str(v))
            f.write(k + ' : ' + str(v) + '\n')

    # mp.set_start_method(platform.python_version()[0] == '3' and 'spawn' or 'fork')  # Force true spawning (not forking) if available
    # For the FigintICE enviroment, can only under Fork model, spawn model will crash the game in the single machine. One soluation is to train async on different machine

    T = Counter()  # Global Epshared counter
    BEST = Counter()
    BEST.set(-9999)
    pre_best = BEST.value()
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Create model for both learner and actor
    env = gym.make(args.env, java_env_path=".", port=args.port, p2=args.p2)
    memory = EpisodicReplayMemory(args.memory_capacity, args.max_episode_length)
    model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
    shared_model = copy.deepcopy(model)
    average_model = copy.deepcopy(model)
    shared_average_model = copy.deepcopy(model)
    model.to(device)
    average_model.to(device)
    optimiser = SharedRMSprop(shared_model.parameters(), lr=args.lr, alpha=args.rmsprop_decay)
    env.close()
    scores = []
    m_scores = []
    del env

    if args.model and os.path.isfile(args.model):
        # Load pretrained weights
        print("Load model from checkpoint {}".format(args.model))
        model.load_state_dict(torch.load(args.model))
        average_model.load_state_dict(torch.load(args.model))
        shared_model.load_state_dict(torch.load(args.model, map_location="cpu"))
        shared_average_model.load_state_dict(torch.load(args.model, map_location="cpu"))
    if args.memory and os.path.isdir(args.memory):
        memory.load(args.memory)
        print("Load memory from CheckPoint {}, memory len: {}".format(args.memory, len(memory)))
    if args.data and os.path.isfile(args.data):
        T.set(torch.load(args.data)[0])
        BEST.set(torch.load(args.data)[1])
        scores = torch.load(args.data)[2]
        m_scores = torch.load(args.data)[3]
        pre_best = BEST.value()
        print("Load data from CheckPoint {}, T: {}.BEST: {}".format(args.data, T.value(), BEST.value()))

    memory_queue = mp.SimpleQueue()
    model_queue = mp.SimpleQueue()
    processes = []
    p2_list = ["ReiwaThunder", "RHEA_PI", "Toothless", "FalzAI"]
    if not args.evaluate:
        # Start training agents
        for rank in range(1, args.num_processes + 1):
            model_queue.put((shared_model.state_dict(), shared_average_model.state_dict()))
            p2 = p2_list[(rank - 1) % len(p2_list)]
            p = mp.Process(target=actor, args=(rank, args, T, BEST, memory_queue, model_queue, p2))
            p.start()
            print('Process ' + str(rank) + ' started')
            processes.append(p)
            sleep(15)

    c_t = 0
    # Learner Loop
    while T.value() <= args.T_max:
        # receive data from actor then train and record into tensorboard
        # if not memory_queue.empty():
        print("Going to read data from ACTOR......")
        received_obj = memory_queue.get()
        print("Finish Reading data from ACTOR!!!!!!")
        on_policy_data = copy.deepcopy(received_obj)
        del received_obj
        T.increment()
        t = T.value()
        best = BEST.value()
        (trajectory, (episode_length, round_score,average_entropy)) = on_policy_data
        memory.append_trajectory(trajectory)
        scores.append(round_score)
        m_score = np.mean(scores[-100:])
        m_scores.append(m_score)
        if m_score * 400 > BEST.value():
            BEST.set(int(m_score * 400))
            best = BEST.value()
        writer.add_scalar("reward/round_score", round_score, t)
        writer.add_scalar("reward/mean_score", m_score, t)
        writer.add_scalar("indicator/entropy", average_entropy, t)
        writer.add_scalar("indicator/episode_length", episode_length, t)
        print("EPISODE: {}, BEST: {}, MEAN_SCORE: {}".format(t, best, m_score))
        off_policy_train(args, t, model, memory, average_model, optimiser, on_policy=True, device=device, writer=writer)
        if memory.length() > args.replay_start:
            off_policy_train(args, t, model, memory, average_model, optimiser, device=device, writer=writer)

        # save the best model
        if best > pre_best:
            print("Save BEST model!")
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'model_{}'.format("BEST")))  # Save model params
            memory.save(os.path.join(save_dir,"memory"))
            torch.save((t, best, scores, m_scores),
                       os.path.join(save_dir, 'indicator_{}'.format("BEST")))  # Save data
            pre_best = best

        # deliver model from learner to actor and save the latest model
        if t % (args.num_processes * 10) == 0 and t > 1 and c_t < t:
            shared_model = copy.deepcopy(model)
            shared_model.to(torch.device("cpu"))
            shared_average_model = copy.deepcopy(average_model)
            shared_average_model.to(torch.device("cpu"))
            shared_model_dict = copy.deepcopy(shared_model.state_dict())
            shared_average_model_dict = copy.deepcopy(shared_average_model.state_dict())
            c_t = t
            for _ in range(args.num_processes):
                model_queue.put((shared_model_dict, shared_average_model_dict),)
            print("Saving LATEST model......")
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'model_{}'.format("LATEST",)))  # Save model params
            memory.save(os.path.join(save_dir,"memory"))
            torch.save((t, best, scores, m_scores),
                       os.path.join(save_dir, 'indicator_{}'.format("LATEST",)))  # Save data
            print("Save LATEST model!!!!!!")
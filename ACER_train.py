# -*- coding: utf-8 -*-
import math
import random
import copy
import gym
import numpy as np
from torch import multiprocessing as mp
import torch
from torch import nn
from torch.distributions import Categorical

from OpenAI.ACER.memory import EpisodicReplayMemory
from OpenAI.ACER.model import ActorCritic
from OpenAI.ACER.utils import state_to_tensor,timeout,TimeoutError


# Knuth's algorithm for generating Poisson samples
def _poisson(lmbd):
    L, k, p = math.exp(-lmbd), 0, 1
    while p > L:
        k += 1
        p *= random.uniform(0, 1)
    return max(k - 1, 0)


# Transfers gradients from thread-specific model to shared model
def _transfer_grads_to_shared_model(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


# Adjusts learning rate
def _adjust_learning_rate(optimiser, lr):
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr


# Updates networks
def _update_networks(args, T, model, shared_model, shared_average_model, loss, optimiser):
    # Zero shared and local grads
    optimiser.zero_grad()
    """
    Calculate gradients for gradient descent on loss functions
    Note that math comments follow the paper, which is formulated for gradient ascent
    """
    loss.backward()
    # Gradient L2 normalisation
    nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)

    # Transfer gradients to shared model and update
    # _transfer_grads_to_shared_model(model, shared_model)
    optimiser.step()
    if args.lr_decay:
        # Linearly decay learning rate
        _adjust_learning_rate(optimiser, max(args.lr * (args.T_max - T.value()) / args.T_max, 1e-32))

    # Update shared_average_model
    for model, shared_average_param in zip(model.parameters(), shared_average_model.parameters()):
        shared_average_param = args.trust_region_decay * shared_average_param + (
                1 - args.trust_region_decay) * model


# Computes an "efficient trust region" loss (policy head only) based on an existing loss and two distributions
def _trust_region_loss(model, distribution, ref_distribution, loss, threshold, g, k,device):
    kl = - (ref_distribution * (distribution.log() - ref_distribution.log())).sum(1).mean(0)

    # Compute dot products of gradients
    k_dot_g = (k * g).sum(1).mean(0)
    k_dot_k = (k ** 2).sum(1).mean(0)
    # Compute trust region update
    if k_dot_k.item() > 0:
        trust_factor = ((k_dot_g - threshold) / k_dot_k).clamp(min=0).detach().to(device)
    else:
        trust_factor = torch.zeros(1).to(device)
    # z* = g - max(0, (k^T∙g - δ) / ||k||^2_2)∙k
    trust_loss = loss + trust_factor * kl

    return trust_loss


# Trains model
def _train(args, T, model, shared_average_model, optimiser, policies, Qs, Vs, actions, rewards, Qret,
           average_policies, old_policies=None,device=None, writer=None):
    off_policy = old_policies is not None
    action_size = policies[0].size(1)
    policy_loss, value_loss = 0, 0

    # Calculate n-step returns in forward view, stepping backwards from the last state
    t = len(rewards)
    for i in reversed(range(t)):
        # Importance sampling weights ρ ← π(∙|s_i) / µ(∙|s_i); 1 for on-policy
        if off_policy:
            rho = policies[i].detach() / (old_policies[i] + 1e-20)
        else:
            rho = torch.ones(1, action_size)
        rho = rho.to(device)
        # Qret ← r_i + γQret
        Qret = rewards[i] + args.discount * Qret
        # Advantage A ← Qret - V(s_i; θ)
        A = Qret - Vs[i]

        # Log policy log(π(a_i|s_i; θ))
        log_prob = (policies[i] + 1e-20).gather(1, actions[i]).log()
        # g ← min(c, ρ_a_i)∙∇θ∙log(π(a_i|s_i; θ))∙A
        single_step_policy_loss = -(rho.gather(1, actions[i]).clamp(max=args.trace_max) * log_prob * A.detach()).mean(
            0)  # Average over batch
        # Off-policy bias correction
        if off_policy:
            # g ← g + Σ_a [1 - c/ρ_a]_+∙π(a|s_i; θ)∙∇θ∙log(π(a|s_i; θ))∙(Q(s_i, a; θ) - V(s_i; θ)
            bias_weight = (1 - args.trace_max / rho).clamp(min=0) * policies[i]
            single_step_policy_loss -= (
                    bias_weight * (policies[i] + 1e-20).log() * (Qs[i].detach() - Vs[i].expand_as(Qs[i]).detach())).sum(
                1).mean(0)
        if args.trust_region:
            # KL divergence k ← ∇θ0∙DKL[π(∙|s_i; θ_a) || π(∙|s_i; θ)]
            k = -average_policies[i].gather(1, actions[i]) / (policies[i].gather(1, actions[i]) + 1e-20)
            if off_policy:
                g = (rho.gather(1, actions[i]).clamp(max=args.trace_max) * A / (policies[i] + 1e-20).gather(1,
                                                                                                            actions[i]) \
                     + (bias_weight * (Qs[i] - Vs[i].expand_as(Qs[i])) / (policies[i] + 1e-20)).sum(1)).detach()
            else:
                g = (rho.gather(1, actions[i]).clamp(max=args.trace_max) * A / (policies[i] + 1e-20).gather(1, actions[
                    i])).detach()
            # Policy update dθ ← dθ + ∂θ/∂θ∙z*
            policy_loss += _trust_region_loss(model, policies[i].gather(1, actions[i]) + 1e-20,
                                              average_policies[i].gather(1, actions[i]) + 1e-20,
                                              single_step_policy_loss, args.trust_region_threshold, g, k, device=device)
        else:
            # Policy update dθ ← dθ + ∂θ/∂θ∙g
            policy_loss += single_step_policy_loss

        # Entropy regularisation dθ ← dθ + β∙∇θH(π(s_i; θ))
        # Sum over probabilities, average over batch
        entropy_regular = args.entropy_weight * -((policies[i] + 1e-20).log() * policies[i]).sum(1).mean(0)
        policy_loss -= entropy_regular
        if np.isnan(policy_loss.item()):
            print("Problem")

        # Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2
        Q = Qs[i].gather(1, actions[i])
        value_loss += ((Qret - Q) ** 2 / 2).mean(0)  # Least squares loss

        # Truncated importance weight ρ¯_a_i = min(1, ρ_a_i)
        truncated_rho = rho.gather(1, actions[i]).clamp(max=1)
        # Qret ← ρ¯_a_i∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
        Qret = truncated_rho * (Qret - Q.detach()) + Vs[i].detach()

    # Zero shared and local grads
    loss = policy_loss + value_loss
    optimiser.zero_grad()
    """
    Calculate gradients for gradient descent on loss functions
    Note that math comments follow the paper, which is formulated for gradient ascent
    """
    loss.backward()
    # Gradient L2 normalisation
    nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
    optimiser.step()
    lr = args.lr
    if args.lr_decay:
        # Linearly decay learning rate
        for param_group in optimiser.param_groups:
            param_group['lr'] = max(args.lr
                                    * (0.5 ** (T / 2000)), args.lr_min)
            lr = param_group['lr']

    # Update shared_average_model
    for model_param, shared_average_param in zip(model.parameters(), shared_average_model.parameters()):
        shared_average_param.data = args.trust_region_decay * shared_average_param.data + (
                1 - args.trust_region_decay) * model_param.data

    if writer is not None:
        writer.add_scalar("loss/policy_loss", policy_loss, T)
        writer.add_scalar("loss/value_loss", value_loss, T)
        writer.add_scalar("loss/learning_rate", lr, T)


# Acts and trains model
def actor(rank, args, T, BEST, memory_queue, model_queue,p2):
    mp.set_sharing_strategy('file_system')
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (args.nofile, rlimit[1]))
    torch.manual_seed(args.seed + rank)
    print("Process {} fighting with {}".format(rank, p2))
    env = gym.make(args.env, java_env_path=".", port=args.port + rank * 2, p2=p2)
    env.seed(args.seed + rank)
    model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
    shared_average_model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
    memory = EpisodicReplayMemory(args.num_processes, args.max_episode_length)

    t = 1  # Thread step counter
    done = True  # Start new episode

    while T.value() <= args.T_max:
        # Actor loop
        t_value = T.value()
        discard = False
        round_score = 0
        episode_length = 0
        sum_entropy = 0
        if not model_queue.empty():
            print("Process {} going to load new model at EPISODE {}......".format(rank, t_value))
            received_obj = model_queue.get()
            model_dict, average_model_dict = copy.deepcopy(received_obj)
            model.load_state_dict(model_dict)
            shared_average_model.load_state_dict(average_model_dict)
            print("Process {} finished loading new mode at EPISODE {}!!!!!!".format(rank, t_value))
            del received_obj

        # Reset or pass on hidden state
        if done:
            hx, avg_hx = torch.zeros(1, args.hidden_size), torch.zeros(1, args.hidden_size)
            cx, avg_cx = torch.zeros(1, args.hidden_size), torch.zeros(1, args.hidden_size)
            # Reset environment and done flag
            try:
                with timeout(seconds=30):
                    s = env.reset()
            except TimeoutError:
                print("Time out to reset env")
                env.close()
                continue
            state = state_to_tensor(s)
            action_mask = [[False for _ in range(56)]]
            action_mask = torch.BoolTensor(action_mask)
            done = False
        else:
            # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
            hx = hx.detach()
            cx = cx.detach()

        # Lists of outputs for training
        policies, Qs, Vs, actions, rewards, average_policies = [], [], [], [], [], []

        while not done:
            # Calculate policy and values
            policy, Q, V, (hx, cx) = model(state, (hx, cx), action_mask)
            average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(state, (avg_hx, avg_cx), action_mask)

            # Sample action
            action = torch.multinomial(policy, 1)[0, 0]
            sum_entropy += Categorical(probs=policy.detach()).entropy()

            # Step
            next_state, reward, done, info = env.step(action.item())
            valid_actions = info.get('my_action_enough', {})
            # get valid actions
            if len(valid_actions) > 0:
                action_mask = [[False if i in valid_actions else True for i in range(56)]]
            else:
                action_mask = [[False for _ in range(56)]]
            action_mask = torch.BoolTensor(action_mask)
            round_score += reward
            if info.get('no_data_receive', False):
                env.close()
                discard = True
                memory.append_transition(state, None, None, None, action_mask.detach(), discard=discard)
                break
            next_state = state_to_tensor(next_state)
            reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards

            # Save (beginning part of) transition for offline training
            memory.append_transition(state, action, reward, policy.detach(), action_mask.detach())  # Save just tensors
            [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies),
                                               (
                                                   policy, Q, V, torch.LongTensor([[action]]), torch.Tensor([[reward]]),
                                                   average_policy))]

            # Increment counters
            t += 1
            episode_length += 1  # Increase episode counter

            # Update state
            state = next_state

        if discard:
            done = True
            continue

        # Finish on-policy episode
        # Do not need to increate T in actor
        # T.increment()
        print("""Process: {}, EPISODE: {},BEST: {}, episode: {}, round_reward: {}""".format(rank, t_value, BEST.value(), t, round_score))

        # Save terminal state for offline training
        memory.append_transition(state, None, None, None, action_mask.detach())
        last_trajectory = copy.deepcopy(memory.last_trajectory())
        on_policy_data = (last_trajectory, (episode_length, round_score, sum_entropy / episode_length))
        send_object = copy.deepcopy(on_policy_data)
        memory_queue.put(send_object,)
        print("Process {} send trajectory".format(rank))
        # TODO: add TD error of the trajectory as the priority
        done = True
    env.close()


def off_policy_train(args, t, model, memory, shared_average_model, optimiser, on_policy=False, device=None, writer=None):
    # Sample a number of off-policy episodes based on the replay ratio
    print("Train the network {}".format("on_policy" if on_policy else "off_policy"))
    n = _poisson(args.replay_ratio) if not on_policy else 1
    for _ in range(n):
        # Act and train off-policy for a batch of (truncated) episode
        if on_policy:
            trajectories = memory.on_policy()
        else:
            trajectories = memory.sample_batch(args.batch_size, maxlen=args.t_max)

        # Reset hidden state
        batch = args.batch_size if not on_policy else 1
        hx = torch.zeros(batch, args.hidden_size).to(device)
        avg_hx = torch.zeros(batch, args.hidden_size).to(device)
        cx = torch.zeros(batch, args.hidden_size).to(device)
        avg_cx = torch.zeros(batch, args.hidden_size).to(device)

        # Lists of outputs for training
        policies, Qs, Vs, actions, rewards, old_policies, average_policies = [], [], [], [], [], [], []

        # Loop over trajectories (bar last timestep)
        for i in range(len(trajectories) - 1):
            # Unpack first half of transition
            state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i]), 0).to(device)
            action = torch.LongTensor([trajectory.action for trajectory in trajectories[i]]).unsqueeze(1).to(device)
            reward = torch.Tensor([trajectory.reward for trajectory in trajectories[i]]).unsqueeze(1).to(device)
            old_policy = torch.cat(tuple(trajectory.policy for trajectory in trajectories[i]), 0).to(device)
            action_mask = torch.cat(tuple(trajectory.action_mask for trajectory in trajectories[i]), 0).to(device)

            # Calculate policy and values
            policy, Q, V, (hx, cx) = model(state, (hx, cx), action_mask)
            average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(state, (avg_hx, avg_cx),action_mask)

            # Save outputs for offline training
            [arr.append(el) for arr, el in
             zip((policies, Qs, Vs, actions, rewards, average_policies, old_policies),
                 (policy, Q, V, action, reward, average_policy, old_policy))]

            # Unpack second half of transition
            next_state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i + 1]), 0).to(device)
            next_action_mask = torch.cat(tuple(trajectory.action_mask for trajectory in trajectories[i + 1]), 0).to(device)
            done = torch.Tensor([trajectory.action is None for trajectory in trajectories[i + 1]]).unsqueeze(1).to(
                device)

        # Do forward pass for all transitions
        _, _, Qret, _ = model(next_state, (hx, cx), next_action_mask)
        # Qret = 0 for terminal s, V(s_i; θ) otherwise
        Qret = ((1 - done) * Qret).detach()

        # Train the network off-policy
        _train(args, t, model, shared_average_model, optimiser, policies, Qs, Vs, actions, rewards,
               Qret, average_policies, old_policies=old_policies if not on_policy else None, device=device,writer=writer)



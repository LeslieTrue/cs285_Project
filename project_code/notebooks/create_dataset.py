import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product               # Cartesian product for iterators
import torch
# allow us to re-use the framework from the src directory
import sys, os
sys.path.append(os.path.abspath(os.path.join('../src/learning')))

import gridworld as W                       # basic grid-world MDPs
import trajectory as T                      # trajectory generation
import optimizer as O                       # stochastic gradient descent optimizer
import solver as S                          # MDP solver (value-iteration)
import plot as P                            # helper-functions for plotting

device = torch.device(0)

def setup_mdp(idx1, idx2, idx3):
    # create our world
    world = W.IcyGridWorld(size=4, p_slip=0.2)

    # set up the reward function
    reward = np.zeros(world.n_states)
    reward[idx1] = 1.0
    reward[idx2] = 0.65
    reward[idx3] = 0.40

    # set up terminal states
    terminal = [15]

    return world, reward, terminal

# set-up the GridWorld Markov Decision Process

def generate_expert_trajectories(world, reward, terminal, n_trajectories):
    # n_trajectories = 200         # the number of "expert" trajectories
    discount = 0.9               # discount for constructing an "expert" policy
    weighting = lambda x: x**50  # down-weight less optimal actions
    start = [0]                  # starting states for the expert

    # compute the value-function
    value = S.value_iteration(world.p_transition, reward, discount)
    
    # create our stochastic policy using the value function
    policy = S.stochastic_policy_from_value(world, value, w=weighting)
    
    # a function that executes our stochastic policy by choosing actions according to it
    policy_exec = T.stochastic_policy_adapter(policy)
    
    # generate trajectories
    tjs = list(T.generate_trajectories(n_trajectories, world, policy_exec, start, terminal))
    
    return tjs, policy

# generate some "expert" trajectories (and its policy for visualization)

# world, reward, terminal = setup_mdp(-1, 4)
# trajectories, expert_policy = generate_expert_trajectories(world, reward, terminal, 100)
# print(trajectories)

# max_t = 0
# for t in trajectories:
#     lenth = len(t.transitions())
#     if lenth > max_t:
#         max_t = lenth
# print(max_t)


def build_batches(trajectories, batch_size = 1, max_trajectory_length = 30):
    end_s = torch.tensor(16)
    end_a = torch.tensor(4)
    states_batch = end_s.repeat((int(len(trajectories)/batch_size),max_trajectory_length))
    action_batch = end_a.repeat((int(len(trajectories)/batch_size),max_trajectory_length))
    # start = 0
    # while start < max_trajectory_length:
    #     idx = int(start/batch_size)
    #     batches = trajectories[start:start + batch_size]
    #     for t in range(batch_size):
    #         states_batch[t][idx] = batches[t]
    traj_num = 0
    # print(states_batch.shape)
    for t in trajectories:
        states_num = 0
        # print(t._t)
        for s in t._t:
            # print(traj_num, states_num)
            k = s[0]
            states_batch[traj_num][states_num] = k
            if states_num != len(t.transitions()):
                action_batch[traj_num][states_num] = t.transitions()[states_num][1]
            states_num += 1
        
        traj_num += 1
    
    # states_batch = states_batch.to(torch.float32).to(device)
    # action_batch = action_batch.to(torch.float32).to(device)
    states_batch = states_batch.to(torch.float32)
    action_batch = action_batch.to(torch.float32)
    return states_batch, action_batch

# sb, ab = build_batches(trajectories, max_trajectory_length = 100)
# print(sb.shape)


def feature_expectation_from_trajectories(features, trajectories):
    n_states, n_features = features.shape

    fe = np.zeros(n_features)

    for t in trajectories:                  # for each trajectory
        for s in t.states():                # for each state in trajectory
            fe += features[s, :]            # sum-up features

    return fe / len(trajectories)           # average over trajectories

def initial_probabilities_from_trajectories(n_states, trajectories):
    p = np.zeros(n_states)

    for t in trajectories:                  # for each trajectory
        p[t.transitions()[0][0]] += 1.0     # increment starting state

    return p / len(trajectories)            # normalize


def compute_expected_svf(p_transition, p_initial, terminal, reward, eps=1e-5):
    n_states, _, n_actions = p_transition.shape
    nonterminal = set(range(n_states)) - set(terminal)  # nonterminal states
    
    # Backward Pass
    # 1. initialize at terminal states
    zs = np.zeros(n_states)                             # zs: state partition function
    zs[terminal] = 1.0

    # 2. perform backward pass
    for _ in range(2 * n_states):                       # longest trajectory: n_states
        # reset action values to zero
        za = np.zeros((n_states, n_actions))            # za: action partition function

        # for each state-action pair
        for s_from, a in product(range(n_states), range(n_actions)):

            # sum over s_to
            for s_to in range(n_states):
                za[s_from, a] += p_transition[s_from, s_to, a] * np.exp(reward[s_from]) * zs[s_to]
        
        # sum over all actions
        zs = za.sum(axis=1)

    # 3. compute local action probabilities
    p_action = za / zs[:, None]

    # Forward Pass
    # 4. initialize with starting probability
    d = np.zeros((n_states, 2 * n_states))              # d: state-visitation frequencies
    d[:, 0] = p_initial

    # 5. iterate for N steps
    for t in range(1, 2 * n_states):                    # longest trajectory: n_states
        
        # for all states
        for s_to in range(n_states):
            
            # sum over nonterminal state-action pairs
            for s_from, a in product(nonterminal, range(n_actions)):
                d[s_to, t] += d[s_from, t-1] * p_action[s_from, a] * p_transition[s_from, s_to, a]

    # 6. sum-up frequencies
    return d.sum(axis=1)


def maxent_irl(p_transition, features, terminal, trajectories, optim, init, eps=1e-4):
    n_states, _, n_actions = p_transition.shape
    _, n_features = features.shape

    # compute feature expectation from trajectories
    e_features = feature_expectation_from_trajectories(features, trajectories)
    
    # compute starting-state probabilities from trajectories
    p_initial = initial_probabilities_from_trajectories(n_states, trajectories)

    # gradient descent optimization
    omega = init(n_features)        # initialize our parameters
    delta = np.inf                  # initialize delta for convergence check

    optim.reset(omega)              # re-start optimizer
    while delta > eps:              # iterate until convergence
        omega_old = omega.copy()

        # compute per-state reward from features
        reward = features.dot(omega)

        # compute gradient of the log-likelihood
        e_svf = compute_expected_svf(p_transition, p_initial, terminal, reward)
        grad = e_features - features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)
        
        # re-compute detla for convergence check
        delta = np.max(np.abs(omega_old - omega))

    # re-compute per-state reward and return
    return features.dot(omega)

def generate_maxent_reward(world, terminal, trajectories):
    # set up features: we use one feature vector per state
    features = W.state_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated stochastic gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward_maxent = maxent_irl(world.p_transition, features, terminal, trajectories, optim, init)

    return reward_maxent


def create_dataset(num_patches):
    k = 2
    for i in range(num_patches):
        # print(i)
        if i%3 == 0:
            k+=1
        world, reward, terminal = setup_mdp(-1, k, np.random.randint(0, 15))
        trajectories, expert_policy = generate_expert_trajectories(world, reward, terminal, 300)
        new_trj = trajectories.copy()
        for t in trajectories:
            if len(t._t) > 100:
                new_trj.remove(t)
        # max_t = 100
        state_patch, action_patch = build_batches(new_trj,  max_trajectory_length = 100)
        reward = generate_maxent_reward(world, terminal, new_trj)
        torch.save(reward, f'datafold4x4/reward{i}.pt')
        torch.save(action_patch, f'datafold4x4/actions{i}.pt')
        torch.save(state_patch, f'datafold4x4/states{i}.pt')


create_dataset(10)


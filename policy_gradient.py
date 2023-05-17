import gym
import torch
import numpy as np
import torch.optim as optim
from collections import deque
import math
import torch.nn as nn
import torch.nn.functional as F
from utils import get_action, log_density, rollout, combine_sample_trajs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(policy, baseline, trajs, policy_optim, baseline_optim, gamma=0.99, baseline_train_batch_size=64, baseline_num_epochs=5):
    # Fill in your policy gradient implementation here

    # TODO: Compute the returns on the current batch of trajectories
    # Hint: Go through all the trajectories in trajs and compute their return to go: discounted sum of rewards from that timestep to the end. 
    # Hint: This is easy to do if you go backwards in time and sum up the reward as a running sum. 
    # Hint: Remember that return to go is return = r[t] + gamma*r[t+1] + gamma^2*r[t+2] + ...

    # TODO: Normalize the returns by subtracting mean and dividing by std
    # Hint: Just do return - return.mean()/ (return.std() + EPS), where EPS is a small constant for numerics

    # TODO: Train baseline by regressing onto returns
    # Hint: Regress the baseline from each state onto the above computed return to go. You can use similar code to behavior cloning to do so. 
    # Hint: Iterate for baseline_num_epochs with batch size = baseline_train_batch_size

    # TODO: Train policy by optimizing surrogate objective: -log prob * (return - baseline)
    # Hint: Policy gradient is given by: \grad log prob(a|s)* (return - baseline)
    # Hint: Return is computed above, you can computer log_probs using the log_density function imported. 
    # Hint: You can predict what the baseline outputs for every state.  
    # Hint: Then simply compute the surrogate objective by taking the objective as -log prob * (return - baseline)
    # Hint: You can then use standard pytorch machinery to take *one* gradient step on the policy
    pass

# Training loop for policy gradient
def simulate_policy_pg(env, policy, baseline, num_epochs=20000, max_path_length=200, pg_batch_size=100, 
                        gamma=0.99, baseline_train_batch_size=64, baseline_num_epochs=5, print_freq=10, render=False):
    policy_optim = optim.Adam(policy.parameters())
    baseline_optim = optim.Adam(baseline.parameters())

    for iter_num in range(num_epochs):
        sample_trajs = []

        # Sampling trajectories
        for it in range(pg_batch_size):
            sample_traj = rollout(env=env, agent=policy, episode_length=max_path_length, agent_name='pg', render=render)
            sample_trajs.append(sample_traj)
        
        # Logging returns occasionally
        if iter_num % print_freq == 0:
            rewards_np = np.mean(np.asarray([traj['rewards'].sum() for traj in sample_trajs]))
            path_length = np.max(np.asarray([traj['rewards'].shape[0] for traj in sample_trajs]))
            print("Episode: {}, reward: {}, max path length: {}".format(iter_num, rewards_np, path_length))

        # Training model
        train_model(policy, baseline, sample_trajs, policy_optim, baseline_optim, gamma=gamma, 
                    baseline_train_batch_size=baseline_train_batch_size, baseline_num_epochs=baseline_num_epochs)

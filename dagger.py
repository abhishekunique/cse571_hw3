import argparse
import torch
import uuid

filename = str(uuid.uuid4())
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from functools import partial

from policy import MakeDeterministic
import gym

from utils import rollout, relabel_action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_dagger(env, policy, expert_paths, expert_policy=None, num_epochs=500, episode_length=50,
                            batch_size=32, num_dagger_iters=10, num_trajs_per_dagger=10, num_test_rollouts=10, render=False):
    
    # TODO: Fill in your dagger implementation here. Make sure to plot your intermediate returns
    
    # Hint: Loop through num_dagger_iters iterations, at each iteration train a policy on the current dataset.
    # Then rollout the policy, use relabel_action to relabel the actions along the trajectory with "expert_policy" and then add this to current dataset
    # Repeat this so the dataset grows with states drawn from the policy, and relabeled actions using the expert

    # Testing final learned policy
    print("Evaluating final learned policy")
    paths = []
    for k in range(num_test_rollouts):
        env.reset()
        path = rollout(
                env,
                policy,
                agent_name='bc',
                episode_length=episode_length,
                render=render)
        paths.append(path)

    mean_return = np.mean(np.array([path['rewards'].sum() for path in paths]))
    std_return = np.std(np.array([path['rewards'].sum() for path in paths]))
    print("Mean of DAgger return is " + str(mean_return))
    print("Std of DAgger return is " + str(std_return))
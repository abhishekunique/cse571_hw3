import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import argparse
from utils import BCPolicy, generate_paths, get_expert_data, RLPolicy, RLBaseline
from bc import simulate_policy_bc
from dagger import simulate_policy_dagger
from policy_gradient import simulate_policy_pg
import pytorch_utils as ptu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='policy_gradient', help='choose task')
    args = parser.parse_args()

    if args.task == 'policy_gradient':
        # Define environment
        env = gym.make("InvertedPendulum-v2")

        # Define policy and value function
        hidden_dim_pol = 64
        hidden_depth_pol = 2
        hidden_dim_baseline = 64
        hidden_depth_baseline = 2
        policy = RLPolicy(env.observation_space.shape[0], env.action_space.shape[0], hidden_dim=hidden_dim_pol, hidden_depth=hidden_depth_pol)
        baseline = RLBaseline(env.observation_space.shape[0], hidden_dim=hidden_dim_baseline, hidden_depth=hidden_depth_baseline)
        policy.to(device)
        baseline.to(device)

        # Training hyperparameters
        num_epochs=100
        max_path_length=200
        pg_batch_size=100
        gamma=0.99
        baseline_train_batch_size=64
        baseline_num_epochs=5
        print_freq=10

        # Train policy gradient
        simulate_policy_pg(env, policy, baseline, num_epochs=num_epochs, max_path_length=max_path_length, pg_batch_size=pg_batch_size, 
                        gamma=gamma, baseline_train_batch_size=baseline_train_batch_size, baseline_num_epochs=baseline_num_epochs, print_freq=print_freq, render=False)

    if args.task == 'behavior_cloning':
        # Define environment
        env = gym.make("Reacher-v2")

        # Define policy
        hidden_dim = 128
        hidden_depth = 2
        obs_size = env.observation_space.shape[0]
        ac_size = env.action_space.shape[0]
        policy = BCPolicy(obs_size, ac_size, hidden_dim=hidden_dim, hidden_depth=hidden_depth) # 10 dimensional latent
        policy.to(device)

        # Get the expert data
        file_path = 'data/expert_data.pkl'
        expert_data = get_expert_data(file_path)

        # Training hyperparameters
        num_test_rollouts = 10
        episode_length = 50
        num_epochs = 200
        batch_size = 32

        # Train behavior cloning
        simulate_policy_bc(env, policy, expert_data, num_epochs=num_epochs, episode_length=episode_length,
                            batch_size=batch_size, num_test_rollouts=num_test_rollouts, render=False)

    if args.task == 'dagger':
        # Define environment
        env = gym.make("Reacher-v2")

        # Policy
        hidden_dim = 1000
        hidden_depth = 3
        obs_size = env.observation_space.shape[0]
        ac_size = env.action_space.shape[0]
        policy = BCPolicy(obs_size, ac_size, hidden_dim=hidden_dim, hidden_depth=hidden_depth) # 10 dimensional latent
        policy.to(device)

        # Get the expert data
        file_path = 'data/expert_data.pkl'
        expert_data = get_expert_data(file_path)

        # Load interactive expert
        expert_policy = torch.load('data/expert_policy.pkl')
        print("Expert policy loaded")
        expert_policy.to(device)
        ptu.set_gpu_mode(True)

        # Training hyperparameters
        num_test_rollouts = 10
        episode_length = 50
        num_epochs = 400
        batch_size = 32
        num_dagger_iters=10
        num_trajs_per_dagger=10
        num_test_rollouts = 10

        # Train DAgger
        simulate_policy_dagger(env, policy, expert_data, expert_policy, num_epochs=num_epochs, episode_length=episode_length,
                               batch_size=batch_size, num_dagger_iters=num_dagger_iters, num_trajs_per_dagger=num_trajs_per_dagger, 
                               num_test_rollouts=num_test_rollouts, render=False)
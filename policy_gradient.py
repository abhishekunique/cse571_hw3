import torch
import numpy as np
import torch.optim as optim
from utils import log_density, rollout

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(policy, baseline, trajs, policy_optim, baseline_optim, gamma=0.99, baseline_train_batch_size=64, baseline_num_epochs=5):
    # Fill in your policy gradient implementation here

    # TODO: Compute the returns on the current batch of trajectories
    # Hint: Go through all the trajectories in trajs and compute their return to go: discounted sum of rewards from that timestep to the end. 
    # Hint: This is easy to do if you go backwards in time and sum up the reward as a running sum. 
    # Hint: Remember that return to go is return = r[t] + gamma*r[t+1] + gamma^2*r[t+2] + ...
    states_all = []
    actions_all = []
    returns_all = []
    for traj in trajs:
        states_singletraj = traj['observations']
        actions_singletraj = traj['actions']
        rewards_singletraj = traj['rewards']
        returns_singletraj = np.zeros_like(rewards_singletraj)
        # TODO start
            
        # TODO end
        states_all.append(states_singletraj)
        actions_all.append(actions_singletraj)
        returns_all.append(returns_singletraj)
    states = np.concatenate(states_all)
    actions = np.concatenate(actions_all)
    returns = np.concatenate(returns_all)
    
    # TODO: Normalize the returns by subtracting mean and dividing by std
    # Hint: Just do return - return.mean()/ (return.std() + EPS), where EPS is a small constant for numerics
    # TODO start
    
    # TODO end
    
    # TODO: Train baseline by regressing onto returns
    # Hint: Regress the baseline from each state onto the above computed return to go. You can use similar code to behavior cloning to do so. 
    # Hint: Iterate for baseline_num_epochs with batch size = baseline_train_batch_size
    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)
    for epoch in range(baseline_num_epochs):
        np.random.shuffle(arr)
        for i in range(n // baseline_train_batch_size):
            batch_index = arr[baseline_train_batch_size * i: baseline_train_batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index).to(device)
            # TODO start         
            
            # TODO end
            baseline_optim.zero_grad()
            loss.backward()
            baseline_optim.step()
            
    # TODO: Train policy by optimizing surrogate objective: -log prob * (return - baseline)
    # Hint: Policy gradient is given by: \grad log prob(a|s)* (return - baseline)
    # Hint: Return is computed above, you can computer log_probs using the log_density function imported. 
    # Hint: You can predict what the baseline outputs for every state.  
    # Hint: Then simply compute the surrogate objective by taking the objective as -log prob * (return - baseline)
    # Hint: You can then use standard pytorch machinery to take *one* gradient step on the policy
    mu, std, logstd = policy(torch.Tensor(states).to(device))
    log_policy = log_density(torch.Tensor(actions).to(device), mu, std, logstd)
    baseline_pred = baseline(torch.from_numpy(states).float().to(device))
    # TODO start
    
    # TODO end
    
    policy_optim.zero_grad()
    loss.backward()
    policy_optim.step()
    
    del states, actions, returns, states_all, actions_all, returns_all

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

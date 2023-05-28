import torch
import torch.nn as nn

import math
import copy
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

class RLPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim=64, hidden_depth=2):
        super(RLPolicy, self).__init__()
        self.trunk = mlp(num_inputs, hidden_dim, num_outputs*2, hidden_depth)

    def forward(self, x):
        outs = self.trunk(x)
        mu, logstd = torch.split(outs, outs.shape[-1] // 2, dim=-1)
        std = torch.exp(logstd)
        return mu, std, logstd

class RLBaseline(nn.Module):
    def __init__(self, num_inputs, hidden_dim=64, hidden_depth=2):
        super(RLBaseline, self).__init__()
        self.trunk = mlp(num_inputs, hidden_dim, 1, hidden_depth)

    def forward(self, x):
        v = self.trunk(x)
        return v

def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.cpu().detach().numpy()
    return action

def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)

# Define the forward model
class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden_dim, action_dim, hidden_depth)

    def forward(self, obs):
        next_pred = self.trunk(obs)
        return next_pred

    def get_action(self, obs, **kwargs):
        obs_t = torch.tensor(obs).float().to(device)
        return self.forward(obs_t).cpu().detach().numpy()

def rollout(
        env,
        agent,
        agent_name, # Should be bc, dagger, pg
        episode_length=math.inf,
        render=False,
):
    # Collect the following data
    raw_obs = []
    raw_next_obs = []
    actions = []
    rewards = []
    dones = []
    images = []

    entropy = None
    log_prob = None
    agent_info = None
    path_length = 0

    o = env.reset()
    if render:
        env.render()

    while path_length < episode_length:
        o_for_agent = o

        if agent_name == 'bc' or agent_name == 'dagger':
            action = agent.get_action(o_for_agent)
        elif agent_name.lower() == 'pg':
            mu, std, _ = agent(torch.Tensor(o_for_agent).unsqueeze(0).to(device))
            action = get_action(mu, std)[0]
        else:
            raise KeyError("Invalid agent name")

        # Step the simulation forward
        next_o, r, done, env_info = env.step(copy.deepcopy(action))
        
        # Render the environment
        if render:
            env.render()

        raw_obs.append(o)
        raw_next_obs.append(next_o)
        actions.append(action)
        rewards.append(r)
        dones.append(done)
        path_length += 1
        if done:
            break
        o = next_o

    # Prepare the items to be returned
    observations = np.array(raw_obs)
    next_observations = np.array(raw_next_obs)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    dones = np.array(dones).reshape(-1, 1)
    
    # Return in the following format
    return dict(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        rewards=rewards,
        dones=np.array(dones).reshape(-1, 1),
        images = np.array(images)
    )

def generate_paths(env, expert_policy, episode_length, num_paths, file_path):
    # Initial data collection
    paths = []
    for j in range(num_paths):
        path = rollout(
            env,
            expert_policy,
            agent_name='bc',
            episode_length=episode_length,
            render=False)
        print("return is " + str(path['rewards'].sum()))
        paths.append(path)

    with open(file_path, 'wb') as fp:
        pickle.dump(paths, fp)
    print('Paths has been save to the file')

def get_expert_data(file_path):
    with open(file_path, 'rb') as fp:
        expert_data = pickle.load(fp)
    print('Imported Expert data successfully')
    return expert_data

def relabel_action(path, expert_policy):
    observation = path['observations']
    expert_action = expert_policy.get_action(observation)
    path['actions'] = expert_action[0]
    return path

def combine_sample_trajs(sample_trajs):
    assert len(sample_trajs) > 0

    my_dict = {k: [] for k in sample_trajs[0]}
    sample_trajs[0].keys()
    for sample_traj in sample_trajs:
        for key, value in sample_traj.items():
            my_dict[key].append(value)
    
    for key, value in my_dict.items():
        my_dict[key] = np.array(value)

    return my_dict

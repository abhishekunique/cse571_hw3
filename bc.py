import torch
import torch.optim as optim
import numpy as np
from utils import rollout
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_bc(env, policy, expert_data, num_epochs=500, episode_length=50, 
                       batch_size=32, num_test_rollouts=10, render=False):
    
    # TODO: Fill in your behavior cloning implementation here

    # Hint: Just flatten your expert dataset and use standard pytorch supervised learning code to train the policy. 
    # Make sure to use the batch size and num_epochs provided
    
    # Testing learned policy
    print("Evaluating learned policy")
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
    print("Mean of BC return is " + str(mean_return))
    print("Std of BC return is " + str(std_return))